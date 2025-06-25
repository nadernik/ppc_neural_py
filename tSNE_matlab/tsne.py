"""
tSNE (t-Distributed Stochastic Neighbor Embedding) Implementation
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Optional, Tuple
import warnings


def tsne(X: np.ndarray, no_dims: int = 2, initial_dims: Optional[int] = None,
         perplexity: float = 30.0, max_iter: int = 1000, 
         learning_rate: float = 200.0) -> np.ndarray:
    """
    Perform tSNE dimensionality reduction
    
    Parameters:
    -----------
    X : np.ndarray
        Input data (n_samples, n_features)
    no_dims : int
        Number of output dimensions
    initial_dims : Optional[int]
        Number of dimensions to use for initial PCA
    perplexity : float
        Perplexity parameter
    max_iter : int
        Maximum number of iterations
    learning_rate : float
        Learning rate
    
    Returns:
    --------
    Y : np.ndarray
        Embedded data (n_samples, no_dims)
    """
    if initial_dims is None:
        initial_dims = min(50, X.shape[1])
    
    # Normalize input data
    X = X - np.mean(X, axis=0)
    X = X / np.std(X, axis=0)
    
    # Perform initial PCA if needed
    if X.shape[1] > initial_dims:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=initial_dims)
        X = pca.fit_transform(X)
    
    # Compute pairwise distances
    D = squareform(pdist(X))
    
    # Compute P-values
    P = x2p(D, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = np.maximum(P, 1e-12)
    
    # Initialize Y randomly
    Y = np.random.randn(X.shape[0], no_dims) * 1e-4
    
    # Perform gradient descent
    dY = np.zeros((X.shape[0], no_dims))
    iY = np.zeros((X.shape[0], no_dims))
    gains = np.ones((X.shape[0], no_dims))
    
    for iter in range(max_iter):
        # Compute Q-values
        sum_Y = np.sum(np.square(Y), axis=1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        np.fill_diagonal(num, 0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        
        # Compute gradients
        PQd = np.expand_dims((P - Q) * num, axis=2)
        for i in range(X.shape[0]):
            dY[i, :] = np.sum(np.tile(PQd[i, :, :], (no_dims, 1)).T * (Y[i, :] - Y), axis=0)
        
        # Perform the update
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < 0.01] = 0.01
        iY = learning_rate * gains * dY
        Y = Y - iY
        Y = Y - np.tile(np.mean(Y, axis=0), (X.shape[0], 1))
        
        # Stop lying about P-values after a while
        if iter == 100:
            P = P / 4
    
    return Y


def x2p(X: np.ndarray, perplexity: float = 30.0, tol: float = 1e-5) -> np.ndarray:
    """
    Convert distances to probabilities
    
    Parameters:
    -----------
    X : np.ndarray
        Distance matrix
    perplexity : float
        Perplexity parameter
    tol : float
        Tolerance for binary search
    
    Returns:
    --------
    P : np.ndarray
        Probability matrix
    """
    n = X.shape[0]
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    log_perp = np.log(perplexity)
    
    for i in range(n):
        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = X[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])
        
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - log_perp
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            
            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - log_perp
            tries += 1
        
        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    
    return P


def Hbeta(D: np.ndarray, beta: float) -> Tuple[float, np.ndarray]:
    """
    Compute the perplexity and the P-row for a specific value of the precision
    
    Parameters:
    -----------
    D : np.ndarray
        Distance vector
    beta : float
        Precision parameter
    
    Returns:
    --------
    H : float
        Entropy
    P : np.ndarray
        Probability vector
    """
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    if sumP == 0:
        H = 0
        P = np.zeros_like(P)
    else:
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
    
    return H, P


def d2p(D: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    """
    Convert distances to probabilities (alternative implementation)
    
    Parameters:
    -----------
    D : np.ndarray
        Distance matrix
    perplexity : float
        Perplexity parameter
    
    Returns:
    --------
    P : np.ndarray
        Probability matrix
    """
    return x2p(D, perplexity)


def tsne_d(X: np.ndarray, no_dims: int = 2, perplexity: float = 30.0) -> np.ndarray:
    """
    Simplified tSNE implementation
    
    Parameters:
    -----------
    X : np.ndarray
        Input data
    no_dims : int
        Number of output dimensions
    perplexity : float
        Perplexity parameter
    
    Returns:
    --------
    Y : np.ndarray
        Embedded data
    """
    return tsne(X, no_dims=no_dims, perplexity=perplexity, max_iter=1000)


def tsne_p(X: np.ndarray, no_dims: int = 2, perplexity: float = 30.0) -> np.ndarray:
    """
    tSNE with probability input
    
    Parameters:
    -----------
    X : np.ndarray
        Input data (already converted to probabilities)
    no_dims : int
        Number of output dimensions
    perplexity : float
        Perplexity parameter
    
    Returns:
    --------
    Y : np.ndarray
        Embedded data
    """
    # This is a simplified version that assumes X is already in probability form
    return tsne(X, no_dims=no_dims, perplexity=perplexity, max_iter=1000) 