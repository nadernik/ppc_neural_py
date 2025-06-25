"""
TDT Data Utilities
Functions for handling TDT (Tucker-Davis Technologies) data
"""

import numpy as np
import scipy.io as sio
import os
from typing import Dict, Any, Optional, Tuple


def add_tdt_event_to_str(rat: int, date: str, tank: str, block: str, 
                         filename: str = 'TDTStruct.mat') -> Dict[str, Any]:
    """
    Add TDT events and snips from a session of recording into a data structure
    
    Parameters:
    -----------
    rat : int
        Rat ID
    date : str
        Recording date
    tank : str
        Tank name
    block : str
        Block name
    filename : str
        Output filename
    
    Returns:
    --------
    tdt_struct : Dict[str, Any]
        TDT data structure
    """
    if not os.path.exists(filename):
        tdt_struct = make_empty_trial_structure()
        sio.savemat(filename, {'TDTStruct': tdt_struct})
    else:
        tdt_struct = sio.loadmat(filename)['TDTStruct']
    
    # Check if this recording session was already added
    if any((tdt_struct['ratID'] == rat) & (tdt_struct['date'] == date)):
        print('This recording session was already added in EVENTS STR.')
        return tdt_struct
    
    # Read TDT events and snips
    tdt_events = tdt2mat_chan(tank, block, event_types=[2, 3], noeneu=True)
    
    # Add new session
    new_day = len(tdt_struct['ratID'])
    new_rd = np.max(tdt_struct['RD']) + 1 if len(tdt_struct['RD']) > 0 else 1
    
    tdt_struct['RD'] = np.append(tdt_struct['RD'], new_rd)
    tdt_struct['ratID'] = np.append(tdt_struct['ratID'], rat)
    tdt_struct['date'] = np.append(tdt_struct['date'], date)
    tdt_struct['events'] = np.append(tdt_struct['events'], tdt_events)
    
    sio.savemat(filename, {'TDTStruct': tdt_struct})
    return tdt_struct


def make_empty_trial_structure() -> Dict[str, Any]:
    """
    Create empty trial structure
    
    Returns:
    --------
    tdt_struct : Dict[str, Any]
        Empty TDT structure
    """
    return {
        'RD': np.array([]),
        'ratID': np.array([]),
        'date': np.array([]),
        'events': np.array([])
    }


def tdt2mat_chan(tank: str, block: str, event_types: Optional[list] = None,
                 noeneu: bool = False) -> Dict[str, Any]:
    """
    Read TDT data from tank and block
    
    Parameters:
    -----------
    tank : str
        Tank name
    block : str
        Block name
    event_types : Optional[list]
        Types of events to read
    noeneu : bool
        Whether to exclude neural data
    
    Returns:
    --------
    tdt_data : Dict[str, Any]
        TDT data structure
    """
    # This is a placeholder implementation
    # In practice, you would use the TDT Python SDK or similar
    print(f"Reading TDT data from tank: {tank}, block: {block}")
    
    # Mock data structure
    tdt_data = {
        'epocs': {},
        'snips': {},
        'streams': {},
        'scalars': {},
        'info': {}
    }
    
    return tdt_data


def extract_spike_times(tdt_data: Dict[str, Any], channel: int) -> np.ndarray:
    """
    Extract spike times from TDT data
    
    Parameters:
    -----------
    tdt_data : Dict[str, Any]
        TDT data structure
    channel : int
        Channel number
    
    Returns:
    --------
    spike_times : np.ndarray
        Array of spike times
    """
    # This is a placeholder implementation
    # In practice, you would extract actual spike times from the TDT data
    print(f"Extracting spike times from channel {channel}")
    
    # Mock spike times
    spike_times = np.random.exponential(100, 1000)  # Random spike times
    spike_times = np.cumsum(spike_times)
    
    return spike_times


def chan2mat(channel_data: np.ndarray, sampling_rate: float = 24414.0625) -> np.ndarray:
    """
    Convert channel data to MATLAB format
    
    Parameters:
    -----------
    channel_data : np.ndarray
        Raw channel data
    sampling_rate : float
        Sampling rate in Hz
    
    Returns:
    --------
    processed_data : np.ndarray
        Processed channel data
    """
    # Basic preprocessing
    processed_data = channel_data.astype(np.float64)
    
    # Remove DC offset
    processed_data = processed_data - np.mean(processed_data)
    
    return processed_data


def save_png(fig, filename: str, dpi: int = 300):
    """
    Save figure as PNG
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename
    dpi : int
        DPI for the saved image
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    print(f"Figure saved as {filename}")


def colormap_custom(name: str = 'redblu') -> np.ndarray:
    """
    Create custom colormap
    
    Parameters:
    -----------
    name : str
        Colormap name
    
    Returns:
    --------
    cmap : np.ndarray
        Colormap array
    """
    if name == 'redblu':
        # Red to blue colormap
        n_colors = 256
        red = np.linspace(1, 0, n_colors)
        blue = np.linspace(0, 1, n_colors)
        green = np.zeros(n_colors)
        cmap = np.column_stack([red, green, blue])
    elif name == 'blues':
        # Blues colormap
        n_colors = 256
        blue = np.linspace(0.5, 1, n_colors)
        red = green = np.linspace(0.5, 1, n_colors)
        cmap = np.column_stack([red, green, blue])
    elif name == 'reds':
        # Reds colormap
        n_colors = 256
        red = np.linspace(0.5, 1, n_colors)
        green = blue = np.linspace(0.5, 1, n_colors)
        cmap = np.column_stack([red, green, blue])
    else:
        # Default to viridis
        from matplotlib import cm
        cmap = cm.viridis(np.linspace(0, 1, 256))
    
    return cmap 