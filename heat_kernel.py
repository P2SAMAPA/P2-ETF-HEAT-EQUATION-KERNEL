import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs

def compute_composite_macro_factor(macro_df):
    """
    Compute composite macro factor as first principal component.
    Returns array of length len(macro_df).
    """
    if macro_df is None or len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    pca.fit(macro_scaled)
    factor = pca.transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def heat_kernel_signature(adj, t=1.0, n_eigen=10):
    """
    Compute heat kernel signature (HKS) for each node.
    HKS(v, t) = Σ exp(-λ_i t) φ_i(v)^2
    where λ_i, φ_i are eigenvalues/eigenvectors of the graph Laplacian.
    """
    # Compute graph Laplacian
    D = np.sum(adj, axis=1)
    L = np.diag(D) - adj
    # Compute eigenvalues and eigenvectors (smallest n_eigen)
    try:
        eigvals, eigvecs = eigs(L, k=n_eigen, which='SM')
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Smallest eigenvalue is zero (connected graph)
        # HKS
        n = adj.shape[0]
        hks = np.zeros(n)
        for i in range(n):
            for k in range(len(eigvals)):
                hks[i] += np.exp(-eigvals[k] * t) * (eigvecs[i, k] ** 2)
        return hks
    except:
        # Fallback: use degree as proxy
        return np.sum(adj, axis=1)

def heat_kernel_score(returns, macro_df, base_time=1.0, time_range=0.5, knn=5):
    """
    Compute per‑ETF HKS with diffusion time adapted to macro.
    """
    n = returns.shape[1]
    if n < 2:
        return np.zeros(n)
    # Build graph from correlation distance (k‑NN)
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    adj = np.zeros((n, n))
    k = min(knn, n-1)
    for i in range(n):
        nearest = np.argsort(dist[i])[1:k+1]
        adj[i, nearest] = 1
    adj = np.maximum(adj, adj.T)
    # Ensure connectivity (no isolated nodes)
    for i in range(n):
        if np.sum(adj[i]) == 0:
            adj[i, (i+1)%n] = 1
            adj[(i+1)%n, i] = 1
    # Compute composite macro factor (last value)
    if macro_df is not None and len(macro_df) > 0:
        macro_factor = compute_composite_macro_factor(macro_df)
        current_macro_factor = macro_factor[-1]
    else:
        current_macro_factor = 0.5
    # Adapt diffusion time: higher macro -> shorter time (faster diffusion)
    diffusion_time = base_time * (1 - current_macro_factor * time_range)
    diffusion_time = max(0.1, min(2.0, diffusion_time))
    # Compute HKS
    hks = heat_kernel_signature(adj, t=diffusion_time)
    return hks
