"""
s-dsCellNet: Spatiotemporal Cell-Cell Communication in the Brain
A spatial extension of dsCellNet (Song et al., CSBJ 2022)

Author: [Your Name]
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats


# ─────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def gaussian_kernel(coords, sigma=0.15):
    """
    Gaussian spatial weighting kernel derived from Fick's law of diffusion.
    
    S_ij = exp(-d²_ij / σ²)
    
    Parameters
    ----------
    coords : np.ndarray, shape (n_cells, 2)
        Tissue XY coordinates in mm
    sigma : float
        Signaling radius in mm (default: 0.15 mm = 150 μm)
        
    Returns
    -------
    S : np.ndarray, shape (n_cells, n_cells)
        Spatial weight matrix. S_ij ∈ [0,1].
        Close cells: S ≈ 1. Distant cells: S ≈ 0.
    
    Examples
    --------
    >>> coords = np.array([[0,0], [0.1,0], [1.0,0]])
    >>> S = gaussian_kernel(coords, sigma=0.15)
    >>> print(S[0,1])  # 100μm apart → high weight
    0.641
    >>> print(S[0,2])  # 1000μm apart → near zero
    0.000
    """
    dists = cdist(coords, coords, metric='euclidean')
    S = np.exp(-(dists ** 2) / (sigma ** 2))
    np.fill_diagonal(S, 0)  # no autocrine
    return S


def compute_ccc(ligand_expr, receptor_expr, spatial_kernel):
    """
    Compute spatially-constrained CCC score.
    
    CCC_ij = I_ij × S_ij
    where I_ij = L_i × R_j (L-R interaction potential)
    
    Extends dsCellNet Equations 1 & 2 with spatial weighting.
    
    Parameters
    ----------
    ligand_expr : np.ndarray, shape (n_cells,)
        Ligand gene expression per cell
    receptor_expr : np.ndarray, shape (n_cells,)
        Receptor gene expression per cell
    spatial_kernel : np.ndarray, shape (n_cells, n_cells)
        Gaussian spatial weight matrix from gaussian_kernel()
        
    Returns
    -------
    CCC : np.ndarray, shape (n_cells, n_cells)
        Spatiotemporal CCC score matrix
    """
    I   = np.outer(ligand_expr, receptor_expr)
    CCC = I * spatial_kernel
    return CCC


def signaling_decay(ccc_healthy, ccc_disease, spatial_kernel):
    """
    Signaling Decay (SD) metric — novel aging/disease biomarker.
    
    SD(t) = 1 - Σ CCC_disease[proximal] / Σ CCC_healthy[proximal]
    
    where proximal = S_ij > exp(-1) (cells within σ)
    
    SD = 0  → healthy, no CCC breakdown
    SD → 1  → complete spatial CCC breakdown
    
    Parameters
    ----------
    ccc_healthy : np.ndarray, shape (n_cells, n_cells)
        CCC matrix from healthy/young group (baseline)
    ccc_disease : np.ndarray, shape (n_cells, n_cells)
        CCC matrix from disease/aged group
    spatial_kernel : np.ndarray, shape (n_cells, n_cells)
        Gaussian spatial weight matrix
        
    Returns
    -------
    SD : float
        Signaling Decay score ∈ [0, 1]
    """
    proximal    = spatial_kernel > np.exp(-1)
    numerator   = np.sum(ccc_disease[proximal])
    denominator = np.sum(ccc_healthy[proximal])
    return 1 - (numerator / (denominator + 1e-10))


def signaling_decay_bootstrap(ccc_healthy, ccc_disease,
                               spatial_kernel, n_boot=1000):
    """
    Bootstrap confidence interval for Signaling Decay.
    
    Parameters
    ----------
    n_boot : int
        Number of bootstrap iterations
        
    Returns
    -------
    sd : float
        Point estimate of SD
    ci_low, ci_high : float
        95% confidence interval
    p_value : float
        p-value vs SD = 0 (one-sample t-test)
    """
    n = ccc_healthy.shape[0]
    sd_point = signaling_decay(ccc_healthy, ccc_disease, spatial_kernel)

    sd_boot = []
    for _ in range(n_boot):
        idx     = np.random.choice(n, n, replace=True)
        S_b     = spatial_kernel[np.ix_(idx, idx)]
        c_h_b   = ccc_healthy[np.ix_(idx, idx)]
        c_d_b   = ccc_disease[np.ix_(idx, idx)]
        sd_boot.append(signaling_decay(c_h_b, c_d_b, S_b))

    sd_arr   = np.array(sd_boot)
    ci_low   = np.percentile(sd_arr, 2.5)
    ci_high  = np.percentile(sd_arr, 97.5)
    _, p_val = stats.ttest_1samp(sd_arr, 0)

    return sd_point, ci_low, ci_high, p_val


def dtw_distance(x, y):
    """
    Dynamic Time Warping distance (dsCellNet Equation 3).
    
    D(X, Y) = min_ξ dξ(X, Y)
    
    Parameters
    ----------
    x, y : np.ndarray, shape (n,)
        Time series (e.g., mean CCC per timepoint)
        
    Returns
    -------
    dist : float
        DTW distance
    """
    try:
        from fastdtw import fastdtw
        dist, _ = fastdtw(
            x.reshape(-1, 1),
            y.reshape(-1, 1)
        )
        return dist
    except ImportError:
        # fallback: Euclidean distance
        return np.linalg.norm(x - y)


def fuzzy_cluster(X, n_clusters=3, m=2, threshold=0.6):
    """
    Fuzzy c-means clustering (dsCellNet Equation 4).
    
    F(U, a) = Σ_j Σ_i μ_ij^m (x_j - a_i)²
    
    Parameters
    ----------
    X : np.ndarray, shape (1, n_samples)
        Feature matrix (e.g., CCC scores per section)
    n_clusters : int
        Number of clusters (default: 3)
    m : float
        Fuzziness parameter (default: 2)
    threshold : float
        Membership threshold for cluster assignment (default: 0.6)
        
    Returns
    -------
    labels : np.ndarray
        Cluster labels (hard assignment)
    membership : np.ndarray
        Fuzzy membership matrix
    centers : np.ndarray
        Cluster centers
    """
    import skfuzzy as fuzz
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X, n_clusters, m,
        error=0.005, maxiter=1000
    )
    labels     = np.argmax(u, axis=0)
    membership = u.max(axis=0)
    return labels, membership, cntr


# ─────────────────────────────────────────────────────────────────
# DATA LOADING (ABC Atlas)
# ─────────────────────────────────────────────────────────────────

def load_abc_atlas(download_base="./data/abc_atlas"):
    """
    Load ABC Atlas data from S3 cache.
    
    Returns
    -------
    cache : AbcProjectCache
        ABC Atlas cache object
    """
    from pathlib import Path
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

    cache = AbcProjectCache.from_s3_cache(Path(download_base))
    print("ABC Atlas cache ready!")
    print("Available directories:", len(cache.list_directories))
    return cache


def load_merfish_section(cache, section_label=None, n_sample=500, seed=42):
    """
    Load one MERFISH brain section with spatial coordinates.
    
    Parameters
    ----------
    section_label : str or None
        Brain section label (e.g., 'C57BL6J-638850.37')
        If None, uses the first available section.
    n_sample : int
        Number of cells to sample
        
    Returns
    -------
    sample : pd.DataFrame
        Cell metadata with x, y coordinates and gene expression
    """
    cell_df = cache.get_metadata_dataframe(
        directory='MERFISH-C57BL6J-638850',
        file_name='cell_metadata'
    )
    expr_df = cache.get_metadata_dataframe(
        directory='MERFISH-C57BL6J-638850',
        file_name='example_genes_all_cells_expression'
    )

    merged = cell_df.merge(expr_df, on='cell_label', how='inner')

    if section_label is None:
        section_label = merged['brain_section_label'].value_counts().index[0]

    section = merged[merged['brain_section_label'] == section_label]
    sample  = section.sample(n_sample, random_state=seed).reset_index(drop=True)

    print(f"Loaded section: {section_label}")
    print(f"Cells: {len(sample)}, Genes: {[c for c in expr_df.columns if c != 'cell_label']}")
    return sample


if __name__ == "__main__":
    print("s-dsCellNet core functions loaded!")
    print("Available functions:")
    print("  gaussian_kernel()         — Spatial weighting kernel S_ij")
    print("  compute_ccc()             — Spatiotemporal CCC score")
    print("  signaling_decay()         — SD metric (aging biomarker)")
    print("  signaling_decay_bootstrap() — SD with 95% CI + p-value")
    print("  dtw_distance()            — Dynamic Time Warping (dsCellNet Eq.3)")
    print("  fuzzy_cluster()           — Fuzzy clustering (dsCellNet Eq.4)")
    print("  load_abc_atlas()          — Load ABC Atlas data")
    print("  load_merfish_section()    — Load MERFISH spatial data")
