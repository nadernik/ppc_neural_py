"""
TDT Data Reader
Functions for reading TDT (Tucker-Davis Technologies) data files
"""

import numpy as np
import os
from typing import Dict, Any, Optional, List, Tuple
import warnings


def tdt2mat(tank: str, block: str, server: str = 'Local', t1: float = 0, 
            t2: float = 0, sort_name: str = 'TankSort', verbose: bool = True,
            data_types: Optional[List[int]] = None, ranges: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Read TDT tank data and convert to Python dictionary
    
    Parameters:
    -----------
    tank : str
        Tank name
    block : str
        Block name
    server : str
        Data tank server
    t1 : float
        Start time for data retrieval
    t2 : float
        End time for data retrieval
    sort_name : str
        Sort ID to use when extracting snippets
    verbose : bool
        Whether to print progress information
    data_types : Optional[List[int]]
        Types of data to retrieve (1: all, 2: epocs, 3: snips, 4: streams, 5: scalars)
    ranges : Optional[np.ndarray]
        Array of valid time range column vectors
    
    Returns:
    --------
    data : Dict[str, Any]
        TDT data structure
    """
    # Initialize data structure
    data = {
        'epocs': {},
        'snips': {},
        'streams': {},
        'scalars': {},
        'info': {}
    }
    
    # Set defaults
    if data_types is None:
        data_types = [1]
    if data_types == [1]:
        data_types = [1, 2, 3, 4]
    
    max_events = int(1e7)
    
    if verbose:
        print(f"\nTank Name:\t{tank}")
        print(f"Block Name:\t{block}")
        print(f"Server:\t\t{server}")
    
    # This is a placeholder implementation
    # In practice, you would use the TDT Python SDK (tdt) or similar
    try:
        # Try to import TDT Python SDK
        import tdt
        print("Using TDT Python SDK")
        
        # Read data using TDT SDK
        data = tdt.read_block(tank, block, t1=t1, t2=t2)
        
    except ImportError:
        print("TDT Python SDK not available, using mock data")
        # Create mock data structure
        data = create_mock_tdt_data(tank, block, verbose)
    
    # Add time ranges if provided
    if ranges is not None:
        data['time_ranges'] = ranges
    
    return data


def create_mock_tdt_data(tank: str, block: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Create mock TDT data for testing purposes
    
    Parameters:
    -----------
    tank : str
        Tank name
    block : str
        Block name
    verbose : bool
        Whether to print information
    
    Returns:
    --------
    data : Dict[str, Any]
        Mock TDT data structure
    """
    data = {
        'epocs': {
            'Tick': {
                'data': np.random.rand(100),
                'onset': np.cumsum(np.random.exponential(1000, 100)),
                'offset': np.cumsum(np.random.exponential(1000, 100)) + 100
            },
            'eLKl': {
                'data': np.random.rand(50),
                'onset': np.cumsum(np.random.exponential(2000, 50)),
                'offset': np.cumsum(np.random.exponential(2000, 50)) + 200
            }
        },
        'snips': {
            'UDPI': {
                'data': np.random.randn(1000, 32),
                'ts': np.cumsum(np.random.exponential(100, 1000)),
                'chan': np.random.randint(1, 33, 1000)
            }
        },
        'streams': {
            'LFP1': {
                'data': np.random.randn(100000),
                'fs': 1000.0
            }
        },
        'scalars': {},
        'info': {
            'tankpath': f'/path/to/{tank}',
            'blockname': block,
            'date': '2024-01-01',
            'starttime': '10:00:00',
            'stoptime': '11:00:00',
            'duration': '01:00:00',
            'durationinSec': 3600.0
        }
    }
    
    if verbose:
        print(f"Tank Path:\t{data['info']['tankpath']}")
        print(f"Start Date:\t{data['info']['date']}")
        print(f"Start Time:\t{data['info']['starttime']}")
        print(f"Stop Time:\t{data['info']['stoptime']}")
        print(f"Total Time:\t{data['info']['duration']}")
    
    return data


def sev2mat(sev_file: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Read SEV file and convert to Python dictionary
    
    Parameters:
    -----------
    sev_file : str
        Path to SEV file
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    data : Dict[str, Any]
        SEV data structure
    """
    if not os.path.exists(sev_file):
        raise FileNotFoundError(f"SEV file not found: {sev_file}")
    
    if verbose:
        print(f"Reading SEV file: {sev_file}")
    
    # This is a placeholder implementation
    # In practice, you would parse the SEV file format
    data = {
        'data': np.random.randn(100000),
        'fs': 24414.0625,
        'nChannels': 1,
        'fileSizeBytes': os.path.getsize(sev_file),
        'dataFormat': 'float32'
    }
    
    return data


def read_sev_quick(sev_file: str) -> np.ndarray:
    """
    Quick read of SEV file (returns only data)
    
    Parameters:
    -----------
    sev_file : str
        Path to SEV file
    
    Returns:
    --------
    data : np.ndarray
        SEV data array
    """
    if not os.path.exists(sev_file):
        raise FileNotFoundError(f"SEV file not found: {sev_file}")
    
    # This is a placeholder implementation
    # In practice, you would read the actual SEV file
    data = np.random.randn(100000)
    
    return data


def plot_chan(data: np.ndarray, fs: float = 24414.0625, title: str = 'Channel Data',
              figsize: Tuple[int, int] = (12, 6)) -> Tuple[Any, Any]:
    """
    Plot channel data
    
    Parameters:
    -----------
    data : np.ndarray
        Channel data
    fs : float
        Sampling frequency
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    import matplotlib.pyplot as plt
    
    time_axis = np.arange(len(data)) / fs
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, data, 'b-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_chan_rwv(data: np.ndarray, fs: float = 24414.0625, title: str = 'Channel Data RWV',
                  figsize: Tuple[int, int] = (12, 6)) -> Tuple[Any, Any]:
    """
    Plot channel data with RWV format
    
    Parameters:
    -----------
    data : np.ndarray
        Channel data
    fs : float
        Sampling frequency
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    import matplotlib.pyplot as plt
    
    time_axis = np.arange(len(data)) / fs
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, data, 'r-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig, ax 