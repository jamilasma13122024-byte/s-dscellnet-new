# s-dsCellNet

**Spatiotemporal Cell-Cell Communication in the Developing and Aging Brain**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-ABC%20Atlas-orange.svg)](https://portal.brain-map.org)

---

## Overview

s-dsCellNet is the **first framework to combine dsCellNet's multi-timepoint temporal CCC analysis** (DTW + fuzzy clustering) with spatially-resolved transcriptomics data.

### Key Results
| Result | Value |
|--------|-------|
| False positive reduction | **99.4%** (174,655 → 374 interactions) |
| Benchmark AUROC | **0.947** vs 0.758 (distance-cutoff) vs 0.506 (non-spatial) |
| SD validation | **r = 0.807**, p < 0.001 (50 brain sections) |
| Aging CCC breakdown | **SD = 0.477**, p < 0.001 |
| GNN improvement | **+14.7%** over non-spatial MLP |

### What We Add to dsCellNet
| Component | Description |
|-----------|-------------|
| Gaussian Kernel | `S_ij = exp(-d²/σ²)` — spatial weighting from Fick's law |
| GNN Encoder | `H^(l+1) = σ(Â·H^(l)·W^(l))` — spatially-aware cell typing |
| CCC Score | `CCC_ij(t) = I_ij(t) × S_ij × h(dz/dt)` — unified score |
| Signaling Decay | `SD(t)` — novel aging biomarker |

---

## Installation

```bash
git clone https://github.com/[username]/s-dsCellNet.git
cd s-dsCellNet
pip install -r requirements.txt
```

---

## Quick Start

```python
import numpy as np
from src.s_dsCellNet import gaussian_kernel, compute_ccc, signaling_decay

# Load your spatial transcriptomics data
coords     = ...  # shape (n_cells, 2) — tissue XY coordinates in mm
ligand     = ...  # shape (n_cells,)   — ligand gene expression
receptor   = ...  # shape (n_cells,)   — receptor gene expression

# Step 1: Gaussian spatial kernel (σ = 150 μm = 0.15 mm)
S = gaussian_kernel(coords, sigma=0.15)

# Step 2: CCC score
CCC = compute_ccc(ligand, receptor, S)

print(f"Spatially-constrained interactions: {(CCC > 0.5).sum():,}")

# Step 3: Signaling Decay (for aging/disease analysis)
# ccc_healthy = CCC from healthy/young group
# ccc_disease = CCC from disease/aged group
SD = signaling_decay(ccc_healthy, ccc_disease, S)
print(f"Signaling Decay: {SD:.4f}")
```

---

## Data

All data used in this study are **publicly available**:

| Dataset | Source | Access |
|---------|--------|--------|
| ABC Atlas MERFISH | Allen Brain Cell Atlas | [portal.brain-map.org](https://portal.brain-map.org) |
| Zeng-Aging-Mouse-10Xv3 | Allen Brain Cell Atlas | [portal.brain-map.org](https://portal.brain-map.org) |

Download data automatically:
```python
from pathlib import Path
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

cache = AbcProjectCache.from_s3_cache(Path("./data/abc_atlas"))
```

---

## Repository Structure

```
s-dsCellNet/
├── src/
│   └── s_dsCellNet.py      # Core functions
├── notebooks/
│   └── tutorial.ipynb      # Step-by-step tutorial
├── figures/
│   ├── figure1_spatial_kernel.png
│   ├── figure2_celltype_ccc.png
│   ├── figure3_signaling_decay.png
│   ├── figure4_dtw_aging.png
│   ├── figure5_benchmark.png
│   ├── figure6_sd_validation.png
│   ├── figure7_fuzzy_clustering.png
│   └── figure8_gnn_result.png
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Methods

### 1. Gaussian Spatial Kernel
Derived from Fick's law of diffusion:
```
S_ij = exp(-d²_ij / σ²)
```
- `d_ij` = Euclidean distance between cells i and j (mm)
- `σ` = signaling radius (default: 0.15 mm = 150 μm)

### 2. GNN Encoder
Spatially-aware cell type deconvolution:
```
H^(l+1) = σ(Â · H^(l) · W^(l))
```
- k=6 nearest neighbour spatial graph
- 2 GCNConv layers (128 → 64 units)

### 3. Signaling Decay
Novel aging biomarker:
```
SD(t) = 1 - Σ CCC_ij(t)·[d<σ] / Σ CCC_ij(0)·[d<σ]
```
- SD = 0: healthy
- SD → 1: complete CCC breakdown

---

## Figures

| Figure | Description |
|--------|-------------|
| Figure 1 | Spatial kernel: 99.4% false positive reduction |
| Figure 2 | Cell type CCC heatmap in tissue space |
| Figure 3 | Signaling Decay across disease stages |
| Figure 4 | DTW multi-timepoint aging analysis |
| Figure 5 | Benchmark: AUROC 0.947 vs competitors |
| Figure 6 | SD validation across 50 brain sections |
| Figure 7 | Fuzzy clustering of CCC patterns |
| Figure 8 | GNN encoder: +14.7% accuracy |

---

## Citation

If you use s-dsCellNet, please cite:

```bibtex
@article{s-dsCellNet2025,
  title   = {s-dsCellNet: A Spatially-Aware Extension of dsCellNet for 
             Inferring Spatiotemporal Cell-Cell Communication Networks 
             in the Developing and Aging Brain},
  author  = {[Author Name] and [Supervisor Name]},
  journal = {Bioinformatics},
  year    = {2025},
  note    = {Under review}
}
```

Also cite the original dsCellNet:
```bibtex
@article{Song2022dsCellNet,
  title   = {dsCellNet: A new computational tool to infer cell-cell 
             communication networks in the developing and aging brain},
  author  = {Song, Zhihong and Wang, Ting and Wu, Yan and Fan, Ming and Wu, Haitao},
  journal = {Computational and Structural Biotechnology Journal},
  volume  = {20},
  pages   = {4072--4081},
  year    = {2022}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

[Your Name] — [email@university.edu]

Department of Computational Biology, [University Name]
