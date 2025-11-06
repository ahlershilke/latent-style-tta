# Test-Time Data Augmentation using Latent Style Statistics for Enhanced Domain Generalization

This is the repository for my master thesis project, conducted under the tutelage of the chair for [Explainable Artificial Intelligence (xAI)](https://www.uni-bamberg.de/xai/), University of Bamberg.

The objective of this work was to develop a method for addressing the domain generalization problem under domain shift conditions by explicitly encoding domain-specific style information during training, thereby enabling latent space augmentation at inference time.

This thesis introduces a new method to improve the robustness of deep neural networks when faced with shifts in data distribution by applying data augmentation at test time using latent style statistics. The approach extracts style-related feature statistics from intermediate layers of a pre-trained model and uses these statistics to generate style-transformed variants of each test input. The augmented inputs are then processed by the model and their predictions are aggregated, reducing sensitivity to domain-specific variations. Experiments on standard domain generalization benchmarks demonstrate that this test-time augmentation can improve accuracy on unseen target domains compared to models without augmentation.

## Motivation

Problem: domain shift (e.g., PACS/VLCS).

Core idea: track “style” (feature mean/std) by domain and layer; at test time, re-normalize activations with target domain stats.

What’s new here: multiple modes (single / selective / average), on-the-fly activation rewriting via forward hooks, uncertainty evaluation via accuracy drop curves.

## Environment setup

Tested with **Python 3.10.** I recommend using ```conda``` to manage the environment.

### Create and activate the environment

```bash
conda create -n latent-style-tta python=3.10 -y
conda activate latent-style-tta
```

### Install PyTorch (choose one)

CPU only:
```bash
pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.1.2 torchvision==0.16.2
```

CUDA 12.1:
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2 torchvision==0.16.2
```

CUDA 11.8:
```bash
pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.1.2 torchvision==0.16.2
```
> If you already installed a CUDA build via the PyTorch index, the next step won’t overwrite it. The version is the same, so pip keeps it.

### Install the rest of the requirements

```bash
pip install -r requirements.txt
```


## Quickstart: Tune → Train → (Optional) TTA

This repository supports **automatic hyperparameter tuning (LODO)**, **training**, and (optionally) **test-time augmentation evaluation (TTA)** in a single command.

### 1. Prepare Your Dataset

Your folder structure should look like this (example for PACS):

```bash
/data/
└── PACS/
    ├── art_painting/
    ├── cartoon/
    ├── photo/
    └── sketch/
```

### 2. Run Tuning + Training

From the repo root:

```bash
python run.py \
    --data_root /data \
    --dataset_name PACS \
    --num_trials 40 \
    --num_epochs 60 \
    --domains 4 \
    --use_mixstyle false \
    --extract_stats true
```

> **Important:** setting `--extract_stats` enables style extraction for TTA

This will:

1. Run per-domain hyperparameter tuning (LODO) with Optuna (`--num_trials` per fold).
2. Compute global best parameters across folds and write them to experiments/configs/global_config.yaml.
3. Train models for each LODO fold using the tuned config (train.py).
4. Save:
    - Tuning results → experiments/hp_results/
    - Global config → experiments/configs/global_config.yaml
    - Model checkpoints → experiments/train_results/saved_models/
    - Visualizations & TensorBoard logs → experiments/train_results/

Optional Flags:
- `--skip_tuning` skip tuning, just run training with a given `--hparam_file`.
- `--skip_training` run tuning only (no training).
- `--hparam_file` path/to/config.yaml, manually specify a hyperparameter config file.
- `--seeds 42 7 0` choose your own random seeds (multi-seed training).

### 3. (Optional) Run TTA Evaluation

You can directly attach the TTA phase to the pipeline by running:

```bash
python run.py \
    --data_root /data \
    --dataset_name PACS \
    --num_trials 40 \
    --num_epochs 60 \
    --domains 4 \
    --use_mixstyle false \
    --extract_stats true \
    --run_tta \
    --tta_output_dir experiments/test_results/pacs_run
```
This will load the saved models from experiments/train_results/saved_models/ and run TTA for all modes and seeds.\
Results will be written as .txt and .json to experiments/test_results/pacs_run/.

Flags:
- `--run_tta` run TTA after training.
- `--tta_output_dir <path>` where to store TTA results.

### 4. View results

- TensorBoard logs
```bash
tensorboard --logdir experiments/train_results/logs
```
- Final training results
    - experiments/train_results/final_results.json contains per-seed and averaged metrics.

- TTA results (if run):
    - .txt and .json summaries in experiments/test_results/.../.


## Repository layout (what lives where)

- *_styleextraction.py – StyleStatistics, DomainAwareHook, StyleExtractorManager.
- *_tta.py – CLI and pipeline: loads models, datasets, runs TTA, saves/plots results.
- data/_datasets.py – dataset splits & domain/class names.
- utils/_visualize.py – plots (uncertainty drop curves, t-SNE, etc.).
- models/_resnet.py (or ._resnet) – backbone with style_stats.

## Key concepts

### Style statistics (μ/σ) per domain & layer

What is stored?

For each domain d and ResNet stage layer ∈ {0,1,2,3}, keep:
- mu_dict[str(layer)][d, C]
- sig_dict[str(layer)][d, C]

Updated with EMA during Training (using warmup):
```bash
momentum = self._get_momentum(layer_idx, update_count)  # warms up for first 100 updates
new_mu = m * old_mu + (1 - m) * batch_mu
```

### Modes of combining stats

- `single_k` use only layer k stats.
- `selective_i_j` use multiple selected layers {i, j}.
- `average` interpolate each layer’s channels to 256, then average across all four layers.

## How style extraction works in this code

- Collecting stats during/after training
    - StyleExtractorManager creates one StyleStatistics per mode (single/selective/average).
    - It hooks the last block of each ResNet stage, runs a dummy forward, and transfers the model’s stored style_stats into the per-mode extractors:
    ```bash
    hook = DomainAwareHook(style_stats=extractor, domain_idx=d, layer_idx=layer)
    layer_module.register_forward_hook(hook)
    _ = model(dummy_input, domain_idx=d)  # collects μ/σ into extractor
    ```
- Saving
    - save_style_stats() writes both JSON and PTH for each mode/domain.

## How TTA uses those stats (the hook trick)

At inference, for each target domain t (not equal to the test domain):

1. Register a DomainAwareHook on the chosen layers.
2. Forward the test batch once with hooks active.
3. Hooks compute per-feature map μ/σ and re-normalize:
```bash
feat_mu = output.mean((2,3), keepdim=True)
feat_sig = output.std((2,3), keepdim=True)
normalized = (output - feat_mu) / (feat_sig + 1e-6)
transformed = normalized * sig_from_stats + mu_from_stats
```
4. Softmax predictions are collected per target domain.

## Uncertainty & evaluation outputs

Per-sample metrics across target domains:

- Class-prob variance (mean over classes)
- Per-sample variance
- Drop curves: remove k% most uncertain samples → accuracy curve; compares to random baseline (AUAD).
- Error detection: AUROC from uncertainty vs correctness.

## Reproducibility

- Seeds: set via `--seeds` (also inside SeedManager).
- Deterministic CuDNN toggles for CUDA.
