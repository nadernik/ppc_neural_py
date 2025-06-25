"""
Neurometric Analysis Example
Example implementation of ROC analysis for neuronal responses
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from typing import Tuple, List
import scipy.io as sio


def load_neuro_data(filename: str = 'NeuroData.mat') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load neurophysiological data
    
    Parameters:
    -----------
    filename : str
        Path to the data file
    
    Returns:
    --------
    c_list : np.ndarray
        List of coherence values
    resp_pref : np.ndarray
        Responses for preferred direction (coherence x trials)
    resp_non_pref : np.ndarray
        Responses for non-preferred direction (coherence x trials)
    """
    try:
        data = sio.loadmat(filename)
        c_list = data['cList'].flatten()
        resp_pref = data['respPref']
        resp_non_pref = data['respNonPref']
    except FileNotFoundError:
        print(f"File {filename} not found. Creating mock data.")
        # Create mock data similar to the example
        c_list = np.array([0.008, 0.016, 0.032, 0.064, 0.128])
        n_trials = 60
        
        # Generate mock responses
        resp_pref = np.zeros((len(c_list), n_trials))
        resp_non_pref = np.zeros((len(c_list), n_trials))
        
        for i, coh in enumerate(c_list):
            # Preferred direction: higher mean with coherence
            mean_pref = 30 + 50 * coh
            resp_pref[i, :] = np.random.poisson(mean_pref, n_trials)
            
            # Non-preferred direction: lower mean with coherence
            mean_non_pref = 25 - 20 * coh
            resp_non_pref[i, :] = np.random.poisson(mean_non_pref, n_trials)
    
    return c_list, resp_pref, resp_non_pref


def plot_response_histograms(c_list: np.ndarray, resp_pref: np.ndarray, 
                           resp_non_pref: np.ndarray, figsize: Tuple[int, int] = (8, 10)) -> None:
    """
    Plot histograms of neuronal responses
    
    Parameters:
    -----------
    c_list : np.ndarray
        List of coherence values
    resp_pref : np.ndarray
        Responses for preferred direction
    resp_non_pref : np.ndarray
        Responses for non-preferred direction
    figsize : Tuple[int, int]
        Figure size
    """
    fig, axes = plt.subplots(len(c_list), 1, figsize=figsize)
    if len(c_list) == 1:
        axes = [axes]
    
    for i in range(len(c_list)):
        ax = axes[len(c_list) - i - 1]
        
        # Plot histograms
        ax.hist(resp_pref[i, :], bins=np.linspace(0, 150, 40), 
               alpha=0.7, color='blue', label='Preferred', density=True)
        ax.hist(resp_non_pref[i, :], bins=np.linspace(0, 150, 40), 
               alpha=0.7, color='red', label='Non-preferred', density=True)
        
        ax.set_xlim(0, 150)
        ax.set_title(f'Coherence {100*c_list[i]:.1f}%', fontsize=12)
        
        if i == 0:
            ax.set_xlabel('Response (spikes/trial)', fontsize=12)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def calculate_roc_curve(resp_pref: np.ndarray, resp_non_pref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate ROC curve for given responses
    
    Parameters:
    -----------
    resp_pref : np.ndarray
        Responses for preferred direction
    resp_non_pref : np.ndarray
        Responses for non-preferred direction
    
    Returns:
    --------
    p_fa : np.ndarray
        False alarm rates
    p_hit : np.ndarray
        Hit rates
    auc : float
        Area under the curve
    """
    n_trials = resp_pref.shape[0]
    crit_list = np.arange(0, max(np.max(resp_pref), np.max(resp_non_pref)) + 1)
    
    p_hit = np.zeros(len(crit_list))
    p_fa = np.zeros(len(crit_list))
    
    for i, criterion in enumerate(crit_list):
        p_hit[i] = np.sum(resp_pref > criterion) / n_trials
        p_fa[i] = np.sum(resp_non_pref > criterion) / n_trials
    
    # Calculate area under the curve
    auc = -np.trapz(p_hit, p_fa)
    
    return p_fa, p_hit, auc


def plot_roc_curves(c_list: np.ndarray, resp_pref: np.ndarray, 
                   resp_non_pref: np.ndarray, figsize: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Plot ROC curves for all coherence values
    
    Parameters:
    -----------
    c_list : np.ndarray
        List of coherence values
    resp_pref : np.ndarray
        Responses for preferred direction
    resp_non_pref : np.ndarray
        Responses for non-preferred direction
    figsize : Tuple[int, int]
        Figure size
    
    Returns:
    --------
    auc_values : np.ndarray
        Area under curve values for each coherence
    """
    plt.figure(figsize=figsize)
    
    auc_values = np.zeros(len(c_list))
    colors = plt.cm.viridis(np.linspace(0, 1, len(c_list)))
    
    for i in range(len(c_list)):
        p_fa, p_hit, auc = calculate_roc_curve(resp_pref[i, :], resp_non_pref[i, :])
        auc_values[i] = auc
        
        plt.plot(p_fa, p_hit, '.-', color=colors[i], 
                label=f'{100*c_list[i]:.1f}%', linewidth=2, markersize=8)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Hit Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('square')
    plt.show()
    
    return auc_values


def plot_neurometric_function(c_list: np.ndarray, auc_values: np.ndarray, 
                            figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot neurometric function
    
    Parameters:
    -----------
    c_list : np.ndarray
        List of coherence values
    auc_values : np.ndarray
        Area under curve values
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.semilogx(100 * c_list, auc_values, 'ko-', markersize=10, 
                markerfacecolor='blue', linewidth=2)
    plt.xlim(0.5, 20)
    plt.ylim(0.4, 1.0)
    plt.xlabel('Coherence (%)')
    plt.ylabel('Proportion Correct')
    plt.title('Neurometric Function')
    plt.grid(True, alpha=0.3)
    plt.show()


def simulate_2afc_task(resp_pref: np.ndarray, resp_non_pref: np.ndarray, 
                      n_simulated_trials: int = 100000) -> float:
    """
    Simulate 2AFC task
    
    Parameters:
    -----------
    resp_pref : np.ndarray
        Responses for preferred direction
    resp_non_pref : np.ndarray
        Responses for non-preferred direction
    n_simulated_trials : int
        Number of simulated trials
    
    Returns:
    --------
    p_correct : float
        Proportion of correct responses
    """
    n_trials = len(resp_pref)
    
    # Draw samples from both distributions
    internal_resp_pref = resp_pref[np.random.randint(0, n_trials, n_simulated_trials)]
    internal_resp_non_pref = resp_non_pref[np.random.randint(0, n_trials, n_simulated_trials)]
    
    # Calculate proportion correct
    p_correct = np.sum(internal_resp_pref > internal_resp_non_pref) / n_simulated_trials
    
    return p_correct


def run_neurometric_analysis(filename: str = 'NeuroData.mat') -> None:
    """
    Run complete neurometric analysis
    
    Parameters:
    -----------
    filename : str
        Path to the data file
    """
    print("Loading neurophysiological data...")
    c_list, resp_pref, resp_non_pref = load_neuro_data(filename)
    
    print("Plotting response histograms...")
    plot_response_histograms(c_list, resp_pref, resp_non_pref)
    
    print("Calculating and plotting ROC curves...")
    auc_values = plot_roc_curves(c_list, resp_pref, resp_non_pref)
    
    print("Plotting neurometric function...")
    plot_neurometric_function(c_list, auc_values)
    
    print("Simulating 2AFC task...")
    p_correct_sim = np.zeros(len(c_list))
    for i in range(len(c_list)):
        p_correct_sim[i] = simulate_2afc_task(resp_pref[i, :], resp_non_pref[i, :])
        print(f"Coherence {100*c_list[i]:.1f}%: ROC AUC = {auc_values[i]:.3f}, "
              f"2AFC simulation = {p_correct_sim[i]:.3f}")
    
    # Compare ROC AUC with 2AFC simulation
    plt.figure(figsize=(8, 6))
    plt.semilogx(100 * c_list, auc_values, 'ko-', markersize=10, 
                markerfacecolor='blue', label='ROC AUC', linewidth=2)
    plt.semilogx(100 * c_list, p_correct_sim, 'rs-', markersize=10, 
                markerfacecolor='red', label='2AFC Simulation', linewidth=2)
    plt.xlim(0.5, 20)
    plt.ylim(0.4, 1.0)
    plt.xlabel('Coherence (%)')
    plt.ylabel('Proportion Correct')
    plt.title('Comparison: ROC AUC vs 2AFC Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_neurometric_analysis() 