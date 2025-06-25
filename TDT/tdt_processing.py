"""
TDT Data Processing
Functions for processing and filtering TDT data
"""

import numpy as np
from scipy import signal
from typing import Dict, Any, Optional, Tuple, List
import warnings


def tdt_filter(data: np.ndarray, fs: float, low_freq: Optional[float] = None,
               high_freq: Optional[float] = None, filter_type: str = 'bandpass',
               order: int = 4) -> np.ndarray:
    """
    Filter TDT data using Butterworth filter
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    fs : float
        Sampling frequency
    low_freq : Optional[float]
        Low frequency cutoff
    high_freq : Optional[float]
        High frequency cutoff
    filter_type : str
        Filter type ('lowpass', 'highpass', 'bandpass', 'bandstop')
    order : int
        Filter order
    
    Returns:
    --------
    filtered_data : np.ndarray
        Filtered data
    """
    if low_freq is None and high_freq is None:
        return data
    
    # Design filter
    if filter_type == 'lowpass':
        if high_freq is None:
            raise ValueError("High frequency must be specified for lowpass filter")
        nyquist = fs / 2
        cutoff = high_freq / nyquist
        b, a = signal.butter(order, cutoff, btype='low')
    elif filter_type == 'highpass':
        if low_freq is None:
            raise ValueError("Low frequency must be specified for highpass filter")
        nyquist = fs / 2
        cutoff = low_freq / nyquist
        b, a = signal.butter(order, cutoff, btype='high')
    elif filter_type == 'bandpass':
        if low_freq is None or high_freq is None:
            raise ValueError("Both low and high frequencies must be specified for bandpass filter")
        nyquist = fs / 2
        low_cutoff = low_freq / nyquist
        high_cutoff = high_freq / nyquist
        b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='band')
    elif filter_type == 'bandstop':
        if low_freq is None or high_freq is None:
            raise ValueError("Both low and high frequencies must be specified for bandstop filter")
        nyquist = fs / 2
        low_cutoff = low_freq / nyquist
        high_cutoff = high_freq / nyquist
        b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='bandstop')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply filter
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data


def lick_pattern(lick_times: np.ndarray, event_times: np.ndarray,
                pre: float = -1000, post: float = 2000, bin_size: float = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze lick pattern around events
    
    Parameters:
    -----------
    lick_times : np.ndarray
        Array of lick times
    event_times : np.ndarray
        Array of event times
    pre : float
        Pre-event time window (ms)
    post : float
        Post-event time window (ms)
    bin_size : float
        Bin size for histogram (ms)
    
    Returns:
    --------
    lick_histogram : np.ndarray
        Lick histogram
    time_axis : np.ndarray
        Time axis
    """
    # Create time axis
    time_axis = np.arange(pre, post, bin_size)
    
    # Initialize histogram
    lick_histogram = np.zeros(len(time_axis))
    
    # Count licks in each bin for each trial
    for event_time in event_times:
        for lick_time in lick_times:
            relative_time = lick_time - event_time
            if pre <= relative_time < post:
                bin_idx = int((relative_time - pre) / bin_size)
                if 0 <= bin_idx < len(lick_histogram):
                    lick_histogram[bin_idx] += 1
    
    return lick_histogram, time_axis


def raster_psth(spike_times: np.ndarray, event_times: np.ndarray,
                pre: float = -500, post: float = 1000, bin_size: float = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create raster plot and PSTH from spike times
    
    Parameters:
    -----------
    spike_times : np.ndarray
        Array of spike times
    event_times : np.ndarray
        Array of event times
    pre : float
        Pre-event time window (ms)
    post : float
        Post-event time window (ms)
    bin_size : float
        Bin size for histogram (ms)
    
    Returns:
    --------
    raster_data : np.ndarray
        Raster plot data
    psth : np.ndarray
        PSTH data
    time_axis : np.ndarray
        Time axis
    """
    from .spike_analysis import align_spikes_to, get_psth
    
    # Align spikes to events
    spike_matrix, _ = align_spikes_to(spike_times, event_times, pre, post, bin_size)
    
    # Calculate PSTH
    psth, _ = get_psth(spike_matrix, bin_size, 50, 'gauss')
    
    # Create time axis
    time_axis = np.arange(pre, post, bin_size)
    
    return spike_matrix, psth, time_axis


def ecog_tdt_data2matlab(tank: str, block: str, channels: Optional[List[int]] = None,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Read ECoG data from TDT tank and convert to MATLAB format
    
    Parameters:
    -----------
    tank : str
        Tank name
    block : str
        Block name
    channels : Optional[List[int]]
        List of channels to read
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    data : Dict[str, Any]
        ECoG data structure
    """
    from .tdt_reader import tdt2mat
    
    # Read TDT data
    tdt_data = tdt2mat(tank, block, verbose=verbose)
    
    # Extract ECoG data
    ecog_data = {}
    
    if 'streams' in tdt_data:
        for stream_name, stream_data in tdt_data['streams'].items():
            if 'LFP' in stream_name or 'ECoG' in stream_name:
                ecog_data[stream_name] = {
                    'data': stream_data['data'],
                    'fs': stream_data['fs'],
                    'channels': channels if channels else list(range(stream_data['data'].shape[0]))
                }
    
    # Add metadata
    ecog_data['info'] = tdt_data['info']
    ecog_data['epocs'] = tdt_data['epocs']
    
    return ecog_data


def process_tdt_streams(tdt_data: Dict[str, Any], 
                       filter_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process TDT streams with optional filtering
    
    Parameters:
    -----------
    tdt_data : Dict[str, Any]
        TDT data structure
    filter_params : Optional[Dict[str, Any]]
        Filtering parameters
    
    Returns:
    --------
    processed_data : Dict[str, Any]
        Processed data structure
    """
    processed_data = tdt_data.copy()
    
    if filter_params is None:
        return processed_data
    
    # Process streams
    if 'streams' in tdt_data:
        for stream_name, stream_data in tdt_data['streams'].items():
            if 'data' in stream_data and 'fs' in stream_data:
                # Apply filter if specified
                if 'filter_type' in filter_params:
                    filtered_data = tdt_filter(
                        stream_data['data'],
                        stream_data['fs'],
                        low_freq=filter_params.get('low_freq'),
                        high_freq=filter_params.get('high_freq'),
                        filter_type=filter_params['filter_type'],
                        order=filter_params.get('order', 4)
                    )
                    processed_data['streams'][stream_name]['data'] = filtered_data
                    processed_data['streams'][stream_name]['filtered'] = True
    
    return processed_data


def extract_event_times(tdt_data: Dict[str, Any], event_name: str) -> np.ndarray:
    """
    Extract event times from TDT data
    
    Parameters:
    -----------
    tdt_data : Dict[str, Any]
        TDT data structure
    event_name : str
        Name of the event
    
    Returns:
    --------
    event_times : np.ndarray
        Array of event times
    """
    if 'epocs' not in tdt_data:
        return np.array([])
    
    if event_name not in tdt_data['epocs']:
        print(f"Warning: Event '{event_name}' not found in TDT data")
        return np.array([])
    
    event_data = tdt_data['epocs'][event_name]
    
    if 'onset' in event_data:
        return np.array(event_data['onset'])
    else:
        return np.array([])


def calculate_firing_rate(spike_times: np.ndarray, window_size: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate firing rate using sliding window
    
    Parameters:
    -----------
    spike_times : np.ndarray
        Array of spike times
    window_size : float
        Window size in milliseconds
    
    Returns:
    --------
    time_axis : np.ndarray
        Time axis
    firing_rate : np.ndarray
        Firing rate array
    """
    if len(spike_times) == 0:
        return np.array([]), np.array([])
    
    # Create time axis
    start_time = spike_times[0] - window_size
    end_time = spike_times[-1] + window_size
    time_axis = np.arange(start_time, end_time, window_size / 10)
    
    # Calculate firing rate
    firing_rate = np.zeros(len(time_axis))
    
    for i, t in enumerate(time_axis):
        # Count spikes in window
        window_start = t - window_size / 2
        window_end = t + window_size / 2
        spike_count = np.sum((spike_times >= window_start) & (spike_times < window_end))
        
        # Convert to Hz
        firing_rate[i] = spike_count / (window_size / 1000.0)
    
    return time_axis, firing_rate 