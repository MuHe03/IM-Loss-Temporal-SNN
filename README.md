# Temporal IM-Loss SNN

Temporal spiking neural network experiments comparing non-spiking membrane
readouts with spiking output readouts trained with and without
information-maximization regularization. Randman is the default lightweight
temporal benchmark; SHD is supported as a larger spike-audio benchmark.

## Structure

- `train_temporal.py` - training entry point.
- `eval_temporal.py` - checkpoint evaluation entry point.
- `models/temporal_snn.py` - feed-forward and recurrent LIF classifiers, surrogate spike function, readouts, and temporal IM losses.
- `data/randman.py` - deterministic random-manifold spike dataset.
- `data/shd.py` - SHD event loader and event-to-bin collation.
- `summarize_shd_results.py` - multiseed SHD summary and plot generation.
- `summarize_ff_output_3way.py` - feed-forward output-condition summary and plot generation.
- `summarize_lambda_sweep.py` - output-layer IM coefficient summary and plot generation.
- `scripts/run_temporal_task.py` - cross-platform task launcher.
- `scripts/smoke_temporal.py` - data-free smoke test.
- `scripts/*.sh` - bash wrappers for Linux, macOS, WSL, Git Bash, and Colab.
- `notebooks/colab_run_experiments.ipynb` - Colab launcher for GPU-backed command runs.
- `datasets/SHD/` - SHD metadata plus local dataset files when downloaded.

## Environment

Use a Python version supported by the PyTorch build you install.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

Colab GPU runtimes usually include PyTorch already; run `pip install -r
requirements.txt` after cloning if any dependency is missing.

## Smoke Tests

```bash
python train_temporal.py --help
python eval_temporal.py --help
python scripts/smoke_temporal.py
python scripts/run_temporal_task.py ff_output_3way 0 --dry-run
```

`scripts/smoke_temporal.py` uses synthetic spike tensors and does not require SHD
files.

## SHD Data

Expected local files:

```text
datasets/SHD/shd_train.h5
datasets/SHD/shd_test.h5
```

If the files are compressed, decompress `shd_train.h5.gz` and `shd_test.h5.gz`
before training.

## Training Commands

Randman non-spiking max membrane readout:

```bash
python train_temporal.py --dataset Randman --output_mode nonspiking --readout max_membrane
```

Randman spiking output with spike-count logits:

```bash
python train_temporal.py --dataset Randman --output_mode spiking_count
```

Randman spiking output with output-layer rate regularization:

```bash
python train_temporal.py --dataset Randman --output_mode spiking_count --use_im_loss --im_loss_type rate --im_include_output --im_hidden_weight 0.0
```

Randman spiking output with output-layer threshold regularization:

```bash
python train_temporal.py --dataset Randman --output_mode spiking_count --use_im_loss --im_loss_type threshold --im_include_output --im_hidden_weight 0.0
```

Short CPU check:

```bash
python train_temporal.py --dataset Randman --no_cuda --epochs 1 --max_train_batches 1 --max_eval_batches 1
```

SHD uses the same training entry point with `--dataset SHD` after the HDF5 files
are available.

Each run writes `config.json`, `metrics.csv`, `last.pth`, `best.pth`, and
`metrics.json` in its run directory. `metrics.csv` stores per-epoch train and
validation curves, including activity diagnostics when enabled.

## Task Runner

Dry-run task commands:

```bash
python scripts/run_temporal_task.py temporal_multiseed 0 --dry-run
python scripts/run_temporal_task.py ff_output_3way 5 --dry-run
python scripts/run_temporal_task.py ff_output_threshold_im 0 --dry-run
```

Run a task:

```bash
python scripts/run_temporal_task.py ff_output_3way 0
```

Write run artifacts outside the repository:

```bash
RUNS_ROOT=/content/drive/MyDrive/im_snn_runs python scripts/run_temporal_task.py ff_output_3way 0
```

PowerShell overrides:

```powershell
$env:RUNS_ROOT="C:\path\to\runs"
$env:NO_CUDA="1"
$env:EPOCHS="1"
$env:MAX_TRAIN_BATCHES="1"
$env:MAX_EVAL_BATCHES="1"
python scripts/run_temporal_task.py ff_output_3way 0
```

Bash overrides:

```bash
NO_CUDA=1 EPOCHS=1 MAX_TRAIN_BATCHES=1 MAX_EVAL_BATCHES=1 python scripts/run_temporal_task.py ff_output_3way 0
```

Lambda sweep:

```bash
IM_LAMBDAS=0,0.0001,0.0003,0.001,0.003,0.01 SEEDS=2020,42,123 python scripts/run_temporal_task.py ff_output_lambda_sweep 0
```

## Evaluation

```bash
python eval_temporal.py runs/.../best.pth
python eval_temporal.py runs/.../best.pth --compute_activity --output runs/.../test_eval.json
```

Bash wrapper:

```bash
bash scripts/eval_temporal_local.sh runs/.../best.pth
COMPUTE_ACTIVITY=1 EVAL_OUTPUT=runs/.../test_eval.json bash scripts/eval_temporal_local.sh runs/.../best.pth
```

## Summaries

```bash
python summarize_shd_results.py --output_dir runs/summary
python summarize_ff_output_3way.py --dataset Randman --output_dir runs/summary
python summarize_lambda_sweep.py --dataset Randman --loss_type threshold --output_dir runs/summary
```

Use `--run_root` with the summary scripts when runs were written to a custom
location. Use `--allow_partial` with `summarize_ff_output_3way.py` for quick
one-seed checks.

## Citation

```bibtex
@inproceedings{
guo2022imloss,
title={{IM}-Loss: Information Maximization Loss for Spiking Neural Networks},
author={Yufei Guo and Yuanpei Chen and Liwen Zhang and Xiaode Liu and Yinglei Wang and Xuhui Huang and Zhe Ma},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022}
}
```
