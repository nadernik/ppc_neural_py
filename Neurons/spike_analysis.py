"""
Spike Analysis Functions
copyright Nader Nikbakht, 2014, SISSA - Trieste
"""

import numpy as np
import scipy.io as sio
from scipy import stats
from typing import List, Tuple, Optional, Union
import os


def align_spikes_to(spike_times: np.ndarray, events_to_align_to: np.ndarray, 
                   pre: float, post: float, bin_size: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Aligns spikes to the event vector and generates binary spike matrix
    
    Parameters:
    -----------
    spike_times : np.ndarray
        Array of spike times
    events_to_align_to : np.ndarray
        Array of event times to align spikes to
    pre : float
        Pre-event time window (ms)
    post : float
        Post-event time window (ms)
    bin_size : float
        Bin size for histogram (ms)
    
    Returns:
    --------
    spike_matrix : np.ndarray
        Binary spike matrix (trials x time_bins)
    aligned_spike_times : List[np.ndarray]
        List of aligned spike times for each trial
    """
    edges = np.linspace(pre, post, int((post - pre) / bin_size) + 1)
    trial_count = len(events_to_align_to)
    spike_matrix = np.zeros((trial_count, len(edges) - 1))
    aligned_spike_times = []
    
    for t in range(trial_count):
        # Find spikes within the time window
        mask = (spike_times > events_to_align_to[t] + pre) & (spike_times < events_to_align_to[t] + post)
        trial_spikes = spike_times[mask] - events_to_align_to[t]
        aligned_spike_times.append(trial_spikes)
        
        if len(trial_spikes) > 0:
            # Create histogram
            hist, _ = np.histogram(trial_spikes, bins=edges)
            spike_matrix[t, :] = hist
    
    spike_matrix = spike_matrix.astype(bool)
    return spike_matrix, aligned_spike_times


def get_psth(spike_matrix: np.ndarray, bin_size: float, w: float, 
             kernel: str = 'gauss') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate PSTH (Peri-Stimulus Time Histogram) with smoothing
    
    Parameters:
    -----------
    spike_matrix : np.ndarray
        Binary spike matrix (trials x time_bins)
    bin_size : float
        Bin size in milliseconds
    w : float
        Width of the filter kernel
    kernel : str
        Type of kernel ('gauss', 'halfgauss', 'exp', 'epsp')
    
    Returns:
    --------
    y : np.ndarray
        Smoothed PSTH
    sem : np.ndarray
        Standard error of the mean
    """
    if kernel == 'gauss':
        gauss_width = max(11, int(6 * w + 1))
        x = np.arange(-gauss_width // 2, gauss_width // 2 + 1)
        kern = stats.norm.pdf(x, 0, w)
    elif kernel == 'halfgauss':
        gauss_width = max(11, int(6 * w + 1))
        x = np.arange(-gauss_width // 2, gauss_width // 2 + 1)
        kern = stats.norm.pdf(x, 0, w)
        kern = kern[len(kern) // 2:]  # half the gaussian
    elif kernel == 'exp':
        x = np.arange(-1, 2, bin_size)
        kern = stats.expon.pdf(x, scale=w) * bin_size
    elif kernel == 'epsp':
        tau_falling = 40  # in ms
        tau_rising = 1    # in ms
        t = np.arange(0, 201, bin_size)
        kern = (stats.expon.pdf(t, scale=tau_falling) * 
                (1 - stats.expon.pdf(t, scale=tau_rising)))
        kern = kern / np.sum(kern)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")
    
    # Normalize kernel
    kern = kern / np.sum(kern)
    
    spike_matrix_mean = np.mean(spike_matrix, axis=0)
    
    # Edge effect correction
    edge_corr_factor = np.convolve(np.ones_like(spike_matrix_mean), kern, mode='same')
    
    # Convolve with kernel
    y = np.convolve(spike_matrix_mean, kern, mode='same') / (bin_size * edge_corr_factor)
    
    # Calculate SEM
    spike_density_trial = np.zeros_like(spike_matrix)
    for i in range(spike_matrix.shape[0]):
        spike_density_trial[i, :] = np.convolve(
            spike_matrix[i, :].astype(float), kern, mode='same'
        ) / (bin_size * edge_corr_factor)
    
    sem = np.std(spike_density_trial, axis=0) / np.sqrt(spike_matrix.shape[0])
    
    # Clip very large SEM values
    sem = np.clip(sem, 0, 100)
    
    return y, sem


def get_roc(spike_counts: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate ROC (Receiver Operating Characteristic) curve and AUC
    
    Parameters:
    -----------
    spike_counts : np.ndarray
        Spike counts for each trial
    labels : np.ndarray
        Binary labels (0 or 1) for each trial
    
    Returns:
    --------
    auc : float
        Area under the ROC curve
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    """
    from sklearn.metrics import roc_curve, auc as roc_auc
    
    fpr, tpr, _ = roc_curve(labels, spike_counts)
    auc = roc_auc(fpr, tpr)
    
    return auc, fpr, tpr


def get_selectivity(spike_counts: np.ndarray, conditions: np.ndarray) -> float:
    """
    Calculate selectivity index
    
    Parameters:
    -----------
    spike_counts : np.ndarray
        Spike counts for each trial
    conditions : np.ndarray
        Condition labels for each trial
    
    Returns:
    --------
    selectivity : float
        Selectivity index
    """
    unique_conditions = np.unique(conditions)
    if len(unique_conditions) != 2:
        raise ValueError("Selectivity calculation requires exactly 2 conditions")
    
    # Calculate mean firing rate for each condition
    rates = []
    for condition in unique_conditions:
        mask = conditions == condition
        rates.append(np.mean(spike_counts[mask]))
    
    # Selectivity = (rate_preferred - rate_nonpreferred) / (rate_preferred + rate_nonpreferred)
    selectivity = (max(rates) - min(rates)) / (max(rates) + min(rates))
    
    return selectivity


def get_tuning(spike_counts: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate tuning curve
    
    Parameters:
    -----------
    spike_counts : np.ndarray
        Spike counts for each trial
    angles : np.ndarray
        Angles/stimulus values for each trial
    
    Returns:
    --------
    unique_angles : np.ndarray
        Unique angle values
    mean_rates : np.ndarray
        Mean firing rates for each angle
    """
    unique_angles = np.unique(angles)
    mean_rates = []
    
    for angle in unique_angles:
        mask = angles == angle
        mean_rates.append(np.mean(spike_counts[mask]))
    
    return unique_angles, np.array(mean_rates) 