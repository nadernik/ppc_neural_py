#!/usr/bin/env python3
"""
Demo script for Python Neuron Analysis
Showcases all major functions in the translated codebase
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Neuron_Analysis.Neurons.spike_analysis import (
    align_spikes_to, get_psth, get_roc, get_selectivity, get_tuning
)
from Neuron_Analysis.Neurons.psth_analysis import (
    plot_spike_psth, plot_spike_raster, plot_population_response,
    plot_tuning_curve, plot_roc_curve
)
from Neuron_Analysis.TDT.tdt_reader import tdt2mat, create_mock_tdt_data
from Neuron_Analysis.TDT.tdt_processing import tdt_filter, extract_event_times
from Neuron_Analysis.dPCA_master.dpca import dpca, dpca_plot
from Neuron_Analysis.tSNE_matlab.tsne import tsne
from Neuron_Analysis.NeurometricExample.neurometric_example import (
    run_neurometric_analysis
)


def demo_spike_analysis():
    """Demo spike analysis functions"""
    print("\n=== Spike Analysis Demo ===")
    
    # Generate example data
    np.random.seed(42)
    n_trials = 50
    n_neurons = 3
    
    # Generate spike times for multiple neurons
    spike_times_list = []
    for neuron in range(n_neurons):
        # Different firing rates for each neuron
        rate = 10 + neuron * 5  # Hz
        spike_times = np.cumsum(np.random.exponential(1000/rate, 1000))
        spike_times_list.append(spike_times)
    
    # Generate event times
    event_times = np.arange(1000, 10000, 2000)
    
    print(f"Generated {len(spike_times_list)} neurons with {len(event_times)} events")
    
    # Align spikes to events
    spike_matrices = []
    for i, spike_times in enumerate(spike_times_list):
        spike_matrix, aligned_spikes = align_spikes_to(
            spike_times, event_times, pre=-500, post=1000, bin_size=10
        )
        spike_matrices.append(spike_matrix)
        print(f"Neuron {i+1}: {spike_matrix.shape[0]} trials, {spike_matrix.shape[1]} time bins")
    
    # Calculate PSTH
    psth, sem = get_psth(spike_matrices[0], bin_size=10, w=50, kernel='gauss')
    print(f"PSTH calculated: {len(psth)} time points")
    
    # Calculate ROC for different conditions
    # Simulate two conditions
    condition1 = np.sum(spike_matrices[0][:25, :], axis=1)  # First 25 trials
    condition2 = np.sum(spike_matrices[0][25:, :], axis=1)  # Last 25 trials
    labels = np.concatenate([np.ones(25), np.zeros(25)])
    
    auc, fpr, tpr = get_roc(condition1, labels)
    print(f"ROC AUC: {auc:.3f}")
    
    # Calculate selectivity
    selectivity = get_selectivity(condition1, labels)
    print(f"Selectivity: {selectivity:.3f}")
    
    # Calculate tuning curve
    angles = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315], n_trials)
    unique_angles, mean_rates = get_tuning(condition1, angles)
    print(f"Tuning curve: {len(unique_angles)} angles")


def demo_psth_plotting():
    """Demo PSTH plotting functions"""
    print("\n=== PSTH Plotting Demo ===")
    
    # Generate example data
    np.random.seed(42)
    spike_times = np.cumsum(np.random.exponential(100, 2000))
    event_times = np.arange(1000, 10000, 1500)
    
    print("Generating PSTH plots...")
    
    # Plot PSTH with raster
    fig, axes = plot_spike_psth(
        spike_times, event_times,
        pre=-500, post=1000,
        title="Demo PSTH"
    )
    plt.show()
    
    # Plot spike raster
    fig, ax = plot_spike_raster(
        spike_times, event_times,
        pre=-500, post=1000,
        title="Demo Raster"
    )
    plt.show()
    
    # Generate population data
    n_neurons = 5
    spike_matrices = []
    for i in range(n_neurons):
        spike_times_neuron = np.cumsum(np.random.exponential(100 + i*20, 1500))
        spike_matrix, _ = align_spikes_to(
            spike_times_neuron, event_times, pre=-500, post=1000, bin_size=10
        )
        spike_matrices.append(spike_matrix)
    
    # Plot population response
    time_axis = np.arange(-500, 1000, 10)
    fig, ax = plot_population_response(
        spike_matrices, time_axis,
        neuron_names=[f"Neuron {i+1}" for i in range(n_neurons)],
        title="Population Response"
    )
    plt.show()
    
    # Plot tuning curve
    angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    firing_rates = 20 + 15 * np.cos(np.radians(angles - 90)) + np.random.normal(0, 2, len(angles))
    
    fig, ax = plot_tuning_curve(
        angles, firing_rates,
        title="Direction Tuning"
    )
    plt.show()
    
    # Plot ROC curve
    fig, ax = plot_roc_curve(
        fpr=np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
        tpr=np.array([0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]),
        auc=0.85,
        title="ROC Curve"
    )
    plt.show()


def demo_tdt_processing():
    """Demo TDT data processing functions"""
    print("\n=== TDT Processing Demo ===")
    
    # Create mock TDT data
    tdt_data = create_mock_tdt_data("DemoTank", "Block-1", verbose=False)
    print("Created mock TDT data")
    
    # Extract event times
    event_times = extract_event_times(tdt_data, 'Tick')
    print(f"Extracted {len(event_times)} event times")
    
    # Filter example data
    fs = 1000  # Hz
    t = np.arange(0, 10, 1/fs)  # 10 seconds
    signal_data = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 100 * t) + np.random.normal(0, 0.1, len(t))
    
    # Apply different filters
    filtered_low = tdt_filter(signal_data, fs, high_freq=50, filter_type='lowpass')
    filtered_high = tdt_filter(signal_data, fs, low_freq=50, filter_type='highpass')
    filtered_band = tdt_filter(signal_data, fs, low_freq=5, high_freq=20, filter_type='bandpass')
    
    print("Applied various filters to signal data")
    
    # Plot filtered data
    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(t[:1000], signal_data[:1000])
    plt.title('Original Signal')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 2)
    plt.plot(t[:1000], filtered_low[:1000])
    plt.title('Low-pass Filtered (< 50 Hz)')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 3)
    plt.plot(t[:1000], filtered_high[:1000])
    plt.title('High-pass Filtered (> 50 Hz)')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 4)
    plt.plot(t[:1000], filtered_band[:1000])
    plt.title('Band-pass Filtered (5-20 Hz)')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()


def demo_dpca():
    """Demo dPCA analysis"""
    print("\n=== dPCA Demo ===")
    
    # Generate example data: neurons x conditions x time
    np.random.seed(42)
    n_neurons = 30
    n_conditions = 8
    n_time = 100
    
    # Create structured data with different marginalizations
    X = np.zeros((n_neurons, n_conditions, n_time))
    
    # Add stimulus-related variance
    for i in range(n_neurons):
        for c in range(n_conditions):
            X[i, c, :] = np.sin(2 * np.pi * (c + 1) * np.arange(n_time) / n_time) * (i + 1) * 0.1
    
    # Add time-related variance
    for i in range(n_neurons):
        for c in range(n_conditions):
            X[i, c, :] += np.exp(-(np.arange(n_time) - 50)**2 / 200) * (i + 1) * 0.05
    
    # Add noise
    X += np.random.normal(0, 0.1, X.shape)
    
    print(f"Generated data: {X.shape}")
    
    # Run dPCA
    W, V, which_marg = dpca(X, num_comps=5)
    print(f"dPCA completed: {W.shape[1]} components")
    print(f"Component marginalizations: {which_marg}")
    
    # Plot dPCA results
    dpca_plot(X, W, V, which_marg, plot_type='traces')
    plt.show()
    
    dpca_plot(X, W, V, which_marg, plot_type='scatter')
    plt.show()


def demo_tsne():
    """Demo tSNE dimensionality reduction"""
    print("\n=== tSNE Demo ===")
    
    # Generate example data with clusters
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create 3 clusters
    cluster1 = np.random.normal(0, 1, (n_samples//3, n_features))
    cluster2 = np.random.normal(5, 1, (n_samples//3, n_features))
    cluster3 = np.random.normal(10, 1, (n_samples//3, n_features))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    labels = np.concatenate([
        np.zeros(n_samples//3),
        np.ones(n_samples//3),
        2 * np.ones(n_samples//3)
    ])
    
    print(f"Generated data: {X.shape} with 3 clusters")
    
    # Run tSNE
    Y = tsne(X, no_dims=2, perplexity=30, max_iter=1000)
    print(f"tSNE completed: {Y.shape}")
    
    # Plot results
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green']
    for i in range(3):
        mask = labels == i
        plt.scatter(Y[mask, 0], Y[mask, 1], c=colors[i], 
                   label=f'Cluster {i+1}', alpha=0.7)
    
    plt.title('tSNE Dimensionality Reduction')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def demo_neurometric():
    """Demo neurometric analysis"""
    print("\n=== Neurometric Analysis Demo ===")
    
    print("Running neurometric analysis...")
    run_neurometric_analysis()


def main():
    """Run all demos"""
    print("Python Neuron Analysis Demo")
    print("=" * 50)
    
    try:
        demo_spike_analysis()
        demo_psth_plotting()
        demo_tdt_processing()
        demo_dpca()
        demo_tsne()
        demo_neurometric()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        print("The Python translation includes:")
        print("- Spike analysis and PSTH calculation")
        print("- PSTH plotting and visualization")
        print("- TDT data processing and filtering")
        print("- dPCA for dimensionality reduction")
        print("- tSNE for visualization")
        print("- Neurometric analysis with ROC curves")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Please check that all dependencies are installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main() 