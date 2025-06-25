"""
PSTH Analysis and Plotting Functions
Original code copyright Nader Nikbakht, 2014, SISSA - Trieste
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Union
from .spike_analysis import align_spikes_to, get_psth


def plot_spike_psth(spike_times: np.ndarray, event_times: np.ndarray, 
                   pre: float = -500, post: float = 1000, bin_size: float = 10,
                   kernel_width: float = 50, kernel_type: str = 'gauss',
                   title: str = 'PSTH', figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot PSTH with raster plot
    
    Parameters:
    -----------
    spike_times : np.ndarray
        Array of spike times
    event_times : np.ndarray
        Array of event times to align to
    pre : float
        Pre-event time window (ms)
    post : float
        Post-event time window (ms)
    bin_size : float
        Bin size for histogram (ms)
    kernel_width : float
        Width of smoothing kernel (ms)
    kernel_type : str
        Type of smoothing kernel
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    
    Returns:
    --------
    fig : plt.Figure
        Figure object
    ax : plt.Axes
        Axes object
    """
    # Align spikes
    spike_matrix, aligned_spike_times = align_spikes_to(
        spike_times, event_times, pre, post, bin_size
    )
    
    # Calculate PSTH
    psth, sem = get_psth(spike_matrix, bin_size, kernel_width, kernel_type)
    
    # Create time axis
    time_axis = np.arange(pre, post, bin_size)
    
    # Create figure with subplots
    fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=figsize, 
                                            gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot raster
    for trial_idx, trial_spikes in enumerate(aligned_spike_times):
        if len(trial_spikes) > 0:
            ax_raster.plot(trial_spikes, [trial_idx] * len(trial_spikes), 
                          'k.', markersize=2, alpha=0.7)
    
    ax_raster.set_xlim(pre, post)
    ax_raster.set_ylim(-1, len(aligned_spike_times))
    ax_raster.set_ylabel('Trial')
    ax_raster.set_title(f'{title} - Raster Plot')
    ax_raster.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    
    # Plot PSTH
    ax_psth.plot(time_axis, psth, 'b-', linewidth=2)
    ax_psth.fill_between(time_axis, psth - sem, psth + sem, 
                        alpha=0.3, color='blue')
    ax_psth.set_xlim(pre, post)
    ax_psth.set_xlabel('Time (ms)')
    ax_psth.set_ylabel('Firing Rate (Hz)')
    ax_psth.set_title(f'{title} - PSTH')
    ax_psth.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    ax_psth.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax_raster, ax_psth)


def plot_spike_raster(spike_times: np.ndarray, event_times: np.ndarray,
                     pre: float = -500, post: float = 1000,
                     title: str = 'Spike Raster', figsize: Tuple[int, int] = (10, 8)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spike raster
    
    Parameters:
    -----------
    spike_times : np.ndarray
        Array of spike times
    event_times : np.ndarray
        Array of event times to align to
    pre : float
        Pre-event time window (ms)
    post : float
        Post-event time window (ms)
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    
    Returns:
    --------
    fig : plt.Figure
        Figure object
    ax : plt.Axes
        Axes object
    """
    # Align spikes
    _, aligned_spike_times = align_spikes_to(
        spike_times, event_times, pre, post, 1  # 1ms bin for raster
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raster
    for trial_idx, trial_spikes in enumerate(aligned_spike_times):
        if len(trial_spikes) > 0:
            ax.plot(trial_spikes, [trial_idx] * len(trial_spikes), 
                   'k.', markersize=1, alpha=0.8)
    
    ax.set_xlim(pre, post)
    ax.set_ylim(-1, len(aligned_spike_times))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trial')
    ax.set_title(title)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_population_response(spike_matrices: List[np.ndarray], 
                           time_axis: np.ndarray,
                           neuron_names: Optional[List[str]] = None,
                           title: str = 'Population Response',
                           figsize: Tuple[int, int] = (12, 8)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot population response across multiple neurons
    
    Parameters:
    -----------
    spike_matrices : List[np.ndarray]
        List of spike matrices for each neuron
    time_axis : np.ndarray
        Time axis for plotting
    neuron_names : Optional[List[str]]
        Names of neurons
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    
    Returns:
    --------
    fig : plt.Figure
        Figure object
    ax : plt.Axes
        Axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(spike_matrices)))
    
    for i, spike_matrix in enumerate(spike_matrices):
        # Calculate mean firing rate across trials
        mean_rate = np.mean(spike_matrix, axis=0)
        
        if neuron_names:
            label = neuron_names[i]
        else:
            label = f'Neuron {i+1}'
        
        ax.plot(time_axis, mean_rate, color=colors[i], label=label, linewidth=1.5)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(title)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_tuning_curve(angles: np.ndarray, firing_rates: np.ndarray,
                     sem: Optional[np.ndarray] = None,
                     title: str = 'Tuning Curve',
                     figsize: Tuple[int, int] = (8, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot tuning curve
    
    Parameters:
    -----------
    angles : np.ndarray
        Stimulus angles
    firing_rates : np.ndarray
        Mean firing rates for each angle
    sem : Optional[np.ndarray]
        Standard error of the mean
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    
    Returns:
    --------
    fig : plt.Figure
        Figure object
    ax : plt.Axes
        Axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if sem is not None:
        ax.errorbar(angles, firing_rates, yerr=sem, fmt='o-', 
                   capsize=5, capthick=2, linewidth=2, markersize=8)
    else:
        ax.plot(angles, firing_rates, 'o-', linewidth=2, markersize=8)
    
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show full circle
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    
    return fig, ax


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float,
                  title: str = 'ROC Curve',
                  figsize: Tuple[int, int] = (6, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ROC curve
    
    Parameters:
    -----------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    auc : float
        Area under the curve
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    
    Returns:
    --------
    fig : plt.Figure
        Figure object
    ax : plt.Axes
        Axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig, ax 