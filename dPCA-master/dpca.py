"""
dPCA (demixed Principal Component Analysis) Implementation
Main dPCA algorithm for neural data analysis
"""

import numpy as np
from scipy import linalg
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings


def dpca(X_full: np.ndarray, num_comps: Union[int, List[int]], 
         combined_params: Optional[List] = None, lambda_reg: float = 0.0,
         order: str = 'yes', time_splits: Optional[List[int]] = None,
         time_parameter: Optional[int] = None, not_to_split: Optional[List] = None,
         scale: str = 'no', C_noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform dPCA on the data
    
    Parameters:
    -----------
    X_full : np.ndarray
        Multi-dimensional array of dimensionality D+1, where first dimension 
        corresponds to N neurons and the rest D dimensions to various parameters
    num_comps : Union[int, List[int]]
        Number of dPCA components to extract
    combined_params : Optional[List]
        Cell array specifying which marginalizations should be added together
    lambda_reg : float
        Regularization parameter
    order : str
        Whether to order components by decreasing variance ('yes' or 'no')
    time_splits : Optional[List[int]]
        Array specifying time splits for time period splitting
    time_parameter : Optional[int]
        Time parameter (used with time_splits)
    not_to_split : Optional[List]
        Cell array specifying which marginalizations should NOT be split
    scale : str
        Whether to scale decoders ('yes' or 'no')
    C_noise : Optional[np.ndarray]
        Noise covariance matrix
    
    Returns:
    --------
    W : np.ndarray
        Decoder matrix (N x S)
    V : np.ndarray
        Encoder matrix (N x S)
    which_marg : np.ndarray
        Array indicating which marginalization each component describes
    """
    # Centering
    X = X_full.reshape(X_full.shape[0], -1)
    X = X - np.mean(X, axis=1, keepdims=True)
    X_full_cen = X.reshape(X_full.shape)
    
    # Total variance
    total_var = np.sum(X**2)
    
    # Marginalize
    X_margs, marg_nums = dpca_marginalize(
        X_full_cen, combined_params=combined_params,
        time_splits=time_splits, time_parameter=time_parameter,
        not_to_split=not_to_split, if_flat='yes'
    )
    
    # Initialize
    decoder = []
    encoder = []
    which_marg = []
    
    # Noise covariance
    if C_noise is None:
        C_noise = np.zeros((X.shape[0], X.shape[0]))
    
    # Loop over marginalizations
    for i in range(len(X_margs)):
        if isinstance(num_comps, int):
            nc = num_comps
        else:
            nc = num_comps[marg_nums[i]]
        
        if isinstance(lambda_reg, (int, float)):
            this_lambda = lambda_reg
        else:
            this_lambda = lambda_reg[marg_nums[i]]
        
        if nc == 0:
            continue
        
        # Compute covariance matrix
        try:
            C = X_margs[i] @ X.T @ np.linalg.inv(
                X @ X.T + C_noise + (total_var * this_lambda)**2 * np.eye(X.shape[0])
            )
        except np.linalg.LinAlgError:
            print('Matrix close to singular, using tiny regularization, lambda = 1e-10')
            this_lambda = 1e-10
            C = X_margs[i] @ X.T @ np.linalg.inv(
                X @ X.T + C_noise + (total_var * this_lambda)**2 * np.eye(X.shape[0])
            )
        
        M = C @ X
        eigenvals, eigenvecs = linalg.eigh(M @ M.T)
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvals)[::-1]
        U = eigenvecs[:, idx[:nc]]
        
        P = U
        D = U.T @ C
        
        if scale == 'yes':
            for uu in range(D.shape[0]):
                A = X_margs[i]
                B = P[:, uu:uu+1] @ D[uu:uu+1, :] @ X
                scaling_factor = (A.flatten() @ B.flatten()) / (B.flatten() @ B.flatten())
                D[uu, :] = scaling_factor * D[uu, :]
        
        decoder.append(D)
        encoder.append(P)
        which_marg.extend([i] * nc)
    
    # Concatenate results
    if decoder:
        decoder = np.vstack(decoder)
        encoder = np.hstack(encoder)
        which_marg = np.array(which_marg)
    else:
        decoder = np.array([])
        encoder = np.array([])
        which_marg = np.array([])
    
    # Transpose
    V = encoder
    W = decoder.T
    
    # Flip axes such that all encoders have more positive values
    to_flip = np.sum(np.sign(V), axis=0) < 0
    W[:, to_flip] = -W[:, to_flip]
    V[:, to_flip] = -V[:, to_flip]
    
    # Handle time splits
    if time_splits is not None:
        to_keep = []
        for i in range(max(marg_nums) + 1):
            components = np.where(np.isin(which_marg, np.where(marg_nums == i)[0]))[0]
            
            if len(components) > 0:
                Z = W[:, components].T @ X
                expl_var = np.sum(Z**2, axis=1)
                order_idx = np.argsort(expl_var)[::-1]
                
                if isinstance(num_comps, int):
                    nc = num_comps
                else:
                    nc = num_comps[i]
                
                to_keep.extend(components[order_idx[:nc]])
        
        if to_keep:
            W = W[:, to_keep]
            V = V[:, to_keep]
            which_marg = which_marg[to_keep]
            which_marg = marg_nums[which_marg]
    
    # Order components by explained variance
    if isinstance(num_comps, int) or order == 'yes':
        for i in range(W.shape[1]):
            Z = W[:, i:i+1].T @ X
            expl_var = np.sum(Z**2)
            
            # Find component with maximum explained variance
            max_var = -1
            max_idx = i
            
            for j in range(i, W.shape[1]):
                Z_j = W[:, j:j+1].T @ X
                var_j = np.sum(Z_j**2)
                if var_j > max_var:
                    max_var = var_j
                    max_idx = j
            
            # Swap components if necessary
            if max_idx != i:
                W[:, [i, max_idx]] = W[:, [max_idx, i]]
                V[:, [i, max_idx]] = V[:, [max_idx, i]]
                which_marg[i], which_marg[max_idx] = which_marg[max_idx], which_marg[i]
    
    return W, V, which_marg


def dpca_marginalize(X_full: np.ndarray, combined_params: Optional[List] = None,
                    time_splits: Optional[List[int]] = None, time_parameter: Optional[int] = None,
                    not_to_split: Optional[List] = None, if_flat: str = 'no') -> Tuple[List[np.ndarray], List[int]]:
    """
    Marginalize data for dPCA
    
    Parameters:
    -----------
    X_full : np.ndarray
        Input data
    combined_params : Optional[List]
        Combined parameters specification
    time_splits : Optional[List[int]]
        Time splits
    time_parameter : Optional[int]
        Time parameter
    not_to_split : Optional[List]
        Parameters not to split
    if_flat : str
        Whether to flatten output
    
    Returns:
    --------
    X_margs : List[np.ndarray]
        Marginalized data
    marg_nums : List[int]
        Marginalization numbers
    """
    # This is a simplified implementation
    # In practice, you would implement the full marginalization logic
    
    dims = X_full.shape
    n_neurons = dims[0]
    
    # Simple marginalization: average over all dimensions except neurons
    X_flat = X_full.reshape(n_neurons, -1)
    X_marg = np.mean(X_flat, axis=1, keepdims=True)
    
    return [X_marg], [0]


def dpca_explained_variance(X_full: np.ndarray, W: np.ndarray, V: np.ndarray,
                           which_marg: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate explained variance for dPCA components
    
    Parameters:
    -----------
    X_full : np.ndarray
        Input data
    W : np.ndarray
        Decoder matrix
    V : np.ndarray
        Encoder matrix
    which_marg : np.ndarray
        Marginalization indices
    
    Returns:
    --------
    expl_var : Dict[str, np.ndarray]
        Explained variance for each marginalization
    """
    X = X_full.reshape(X_full.shape[0], -1)
    X = X - np.mean(X, axis=1, keepdims=True)
    
    expl_var = {}
    unique_margs = np.unique(which_marg)
    
    for marg in unique_margs:
        components = which_marg == marg
        if np.any(components):
            W_marg = W[:, components]
            Z = W_marg.T @ X
            expl_var[f'marg_{marg}'] = np.sum(Z**2, axis=1)
    
    return expl_var


def dpca_plot(X_full: np.ndarray, W: np.ndarray, V: np.ndarray, 
              which_marg: np.ndarray, plot_type: str = 'traces') -> None:
    """
    Plot dPCA results
    
    Parameters:
    -----------
    X_full : np.ndarray
        Input data
    W : np.ndarray
        Decoder matrix
    V : np.ndarray
        Encoder matrix
    which_marg : np.ndarray
        Marginalization indices
    plot_type : str
        Type of plot ('traces', 'scatter', 'heatmap')
    """
    import matplotlib.pyplot as plt
    
    X = X_full.reshape(X_full.shape[0], -1)
    X = X - np.mean(X, axis=1, keepdims=True)
    
    Z = W.T @ X
    
    unique_margs = np.unique(which_marg)
    n_margs = len(unique_margs)
    
    if plot_type == 'traces':
        fig, axes = plt.subplots(n_margs, 1, figsize=(10, 3*n_margs))
        if n_margs == 1:
            axes = [axes]
        
        for i, marg in enumerate(unique_margs):
            components = which_marg == marg
            if np.any(components):
                Z_marg = Z[components, :]
                for j in range(Z_marg.shape[0]):
                    axes[i].plot(Z_marg[j, :], label=f'Component {j+1}')
                axes[i].set_title(f'Marginalization {marg}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    elif plot_type == 'scatter':
        if Z.shape[0] >= 2:
            plt.figure(figsize=(10, 8))
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_margs)))
            
            for i, marg in enumerate(unique_margs):
                components = which_marg == marg
                if np.sum(components) >= 2:
                    Z_marg = Z[components, :]
                    plt.scatter(Z_marg[0, :], Z_marg[1, :], 
                              c=[colors[i]], label=f'Marg {marg}', alpha=0.7)
            
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title('dPCA Components')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
    
    elif plot_type == 'heatmap':
        plt.figure(figsize=(12, 8))
        plt.imshow(Z, aspect='auto', cmap='RdBu_r')
        plt.colorbar(label='Component Value')
        plt.xlabel('Time')
        plt.ylabel('Component')
        plt.title('dPCA Components Heatmap')
        plt.show() 