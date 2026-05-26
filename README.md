# PPM-FL
Create environmet with the specs.txt  
The code in ScalableTopologicalRegularizers used for computing PPM is from https://github.com/htwong-ai/ScalableTopologicalRegularizers  
The code in FiltrationLearningForPointClouds for DNN-based methods is from https://github.com/git-westriver/FiltrationLearningForPointClouds  
All experimants are conducted via bash files.

# Reproducibility Guide

This document provides the information needed to reproduce the experiments reported in the paper. We address the reproducibility concerns raised during review by making all entry scripts, utility files, and environment specifications available.

---

## Repository Structure

```
PPM_Expected_FL_2/
├── expected_toporep_2.py          # Core: Expected Topological Representation (PPM)
├── expected_toporep.py            # Earlier version (reference)
├── early_stopping.py              # Early stopping utility used in all experiments
├── utils2.py                      # Data loading and distance matrix sampling utilities
├── specs.txt                      # Exact conda environment (see Environment Setup)
│
├── exp_ModelNet_run_two_phase.py  # ModelNet two-phase experiment (main)
├── exp_ModelNet_run_two_phase_np.py  # ModelNet two-phase with configurable num_points
├── exp_ModelNet_run.py            # ModelNet single-phase experiment
├── exp_protein_run.py             # Protein classification experiment
│
├── exe_ModelNet_two_phase.sh      # → Table: ModelNet classification (two-phase)
├── exe_ModelNet_two_phase_scale.sh   # → Table: Scalability experiment
├── exe_ModelNet_two_phase_robustness.sh  # → Figure: Robustness experiment
├── exe_ModelNet_two_phase_rips.sh    # → Table: Rips baseline comparison
├── exe_ModelNet.sh                   # → Table: ModelNet single-phase baseline
├── exe_protein_all.sh                # → Table: Protein classification
├── exe_protein_d0_rips.sh            # → Table: Protein Rips baseline
├── exe_scale_*.sh                    # → Figure: Scalability by dataset size
│
├── data/                          # Dataset directory (see Data Preparation)
├── premodel/                      # Pre-trained backbone models (deepsets/pointnet/pointmlp)
└── result/                        # Experiment outputs (created automatically)
```

---

## Environment Setup

The full environment is specified in `specs.txt` as a conda explicit package list.

**Key dependencies:**

| Package        | Version  |
|----------------|----------|
| Python         | 3.10     |
| PyTorch        | 2.0.1    |
| CUDA Toolkit   | 11.8     |
| gudhi          | 3.x      |
| numpy          | 1.x      |
| scikit-learn   | 1.x      |

**To recreate the environment:**

```bash
# Option 1: from explicit package list (exact reproduction)
conda create --name Filtration_Learning --file specs.txt

# Option 2: minimal install
conda create -n Filtration_Learning python=3.10
conda activate Filtration_Learning
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install gudhi scikit-learn numpy matplotlib
```

> Note: `specs.txt` was generated on Linux-64 with CUDA 11.8. GPU experiments were run on NVIDIA RTX A6000 (48 GB). CPU-only reproduction is possible but substantially slower.

---

## Data Preparation

### ModelNet-10 (Point Cloud Classification)

The dataset `ModelNetNoisy01_C=10,N=100,T=1,K=2000` contains 10-class ModelNet point clouds with Gaussian noise. Pre-processed tensors are stored under `FiltrationLearningForPointClouds/data/`:

```
FiltrationLearningForPointClouds/data/ModelNetNoisy01_C=10,N=100,T=1,K=2000_data   # shape: (1000, 2000, 3) point clouds
FiltrationLearningForPointClouds/data/ModelNetNoisy01_C=10,N=100,T=1,K=2000_label  # shape: (1000,) class labels
```

These files are already provided. If you need to regenerate them from raw ModelNet10 meshes:

```bash
# Download ModelNet10 from Princeton ModelNet page and extract to:
#   FiltrationLearningForPointClouds/data_create/data/ModelNet10/
cd FiltrationLearningForPointClouds/data_create
python3 create_modelnet_pointcloud_data.py
```

Processing details:
- 100 point clouds per class (10 classes), each sampled with 2000 points from mesh faces
- Coordinates normalized per-cloud: `(x - mean) / std`
- Gaussian noise added: σ = 0.1
- Output stored as PyTorch float32 tensors

### ModelNet-40 (Point Cloud Classification)

The dataset `ModelNetNoisy01_C=40,N=100,T=1,K=2000` is the 40-class variant. Pre-processed tensors:

```
FiltrationLearningForPointClouds/data/ModelNetNoisy01_C=40,N=100,T=1,K=2000_data
FiltrationLearningForPointClouds/data/ModelNetNoisy01_C=40,N=100,T=1,K=2000_label
```

To regenerate from raw ModelNet40 meshes:

```bash
# 1. Download ModelNet40 from Princeton ModelNet page:
#    https://modelnet.cs.princeton.edu/
#    Extract so that the directory structure is:
#    ~/PPM_Expected_FL_2/ModelNet40/<classname>/train/*.off

# 2. Run the creation script:
cd FiltrationLearningForPointClouds/data_create
python3 create_modelnet40_pointcloud_data.py
```

Processing details (from `data_create/create_modelnet40_pointcloud_data.py`):
- 40 classes, 100 point clouds per class selected from training splits
- 2000 points sampled per cloud via area-weighted mesh face sampling
- Coordinates normalized per-cloud: `(x - mean) / max(std, 1e-8)`
- Gaussian noise added: σ = 0.1
- Random seed fixed to 0 for reproducibility of point sampling
- Output stored as PyTorch float32 tensors

To run the ModelNet-40 two-phase experiment:

```bash
bash exe_ModelNet40_two_phase_np.sh
```

Key settings in `exe_ModelNet40_two_phase_np.sh`:

| Parameter   | Value | Meaning |
|-------------|-------|---------|
| `dataset`   | `ModelNetNoisy01_C=40,N=100,T=1,K=2000` | 40-class dataset |
| `dim`       | `-1`  | Use both H₀ and H₁ |
| `nb_repeat` | `200` | Expectation approximation repeats |
| `num_points`| `1024`| Points subsampled per cloud |
| `seed`      | `2026`| Data-split seed |
| `ppm_seed`  | `42`  | PPM subsampling seed |
| `deepsets`  | `1`   | Use DeepSets backbone |

To evaluate the pre-trained DeepSets backbone on ModelNet-40:

```bash
bash exe_eval_ModelNet40_deepsets_np.sh
# Results saved to result/ModelNet40_deepsets_pretrained_eval_np1024_seed2026/eval.log
```

### Protein Classification

Pre-processed tensors are already provided:

```
FiltrationLearningForPointClouds/data/KNproteinNoisy01_C=7,T=500,K=60_data
FiltrationLearningForPointClouds/data/KNproteinNoisy01_C=7,T=500,K=60_label
```

To regenerate from the raw cross-correlation matrices:

```bash
cd FiltrationLearningForPointClouds/data_create
python3 create_KN_protein_sampled_data.py
```


---

## Reproducing Reported Results


### ModelNet Two-Phase Experiments (main table)

```bash
bash exe_ModelNet_two_phase.sh
```

Key hyperparameters set in the script:
- `nb_repeat=200`, `dim=0`, `bs=40`, `deepsets=1`
- 3-fold cross-validation, 200 epochs, early stopping on validation loss (patience=20)



### Robustness Experiments

```bash
bash exe_ModelNet_two_phase_robustness.sh
```

### Rips Filtration Baselines

```bash
bash exe_ModelNet_two_phase_rips.sh
bash exe_protein_d0_rips.sh
```

### Protein Classification

```bash
bash exe_protein_all.sh
```


---

## Early Stopping

All experiments use `early_stopping.py`, which monitors validation loss and saves the best model checkpoint. The handler is initialized in each experiment script as:

```python
from early_stopping import EarlyStopping
early_stopping = EarlyStopping(patience=<p>, verbose=True, path=<checkpoint_dir>, CV_idx=<fold>)
```

The reported test accuracy for each fold is the test accuracy at the epoch with the **lowest validation loss** (i.e., `early_stopping.best_acc`), not the final epoch. This is consistent with the description in the manuscript. We set patience=20 in all of our experiments.

---

## Cross-Validation Protocol

All experiments use 3-fold cross-validation with a fixed random seed:

```python
random.seed(2026)
random.shuffle(data_idx)
# fold k: test on indices where position % 3 == k
```

Final reported accuracy = mean ± std over the 3 folds.

---

## PH Settings Reference

Each experiment script sets the following flags:

| Flag        | Meaning |
|-------------|---------|
| `rips=0`    | Disable Rips baseline backbone |
| `toporep=1` | Enable proposed Expected Topological Representation |
| `dtm=0`     | Disable DTM filtration |
| `dim=0`     | Use H₀ only; `dim=1` for H₁; `dim=-1` for H₀+H₁ |
| `nb_repeat` | Number of random subsets for expectation approximation |
| `deepsets=1`| Use DeepSets as point cloud feature extractor |
| `pointnet=1`| Use PointNet as point cloud feature extractor |
| `pointmlp=1`| Use PointMLP as point cloud feature extractor |

---

## Notes on Reproducibility

- Results may vary slightly across runs due to GPU non-determinism. To fix all seeds, add `torch.manual_seed` and `torch.backends.cudnn.deterministic = True` before training.
- The `specs.txt` file was generated with `conda list --explicit` and can reproduce the exact package versions used in reported experiments.
- All scripts redirect stdout and stderr to `result/<name>/train.log` via `tee` for complete logging.

