# Temporal IM-Loss SNN

## Project Summary

This project tests whether an output-layer information-maximization (IM) regularizer can make spiking output layers competitive with non-spiking membrane readouts in temporal spiking neural networks. We compare feed-forward LIF classifiers on Randman and the Spiking Heidelberg Digits (SHD) dataset using matched architecture, optimizer, random seeds, and training duration. The hypothesis is that output spike quantization can discard useful graded evidence, while IM regularization should increase output spike informativeness and partially reduce that gap, with a trade-off controlled by the IM coefficient lambda. The final results show that readout choice is critical: SHD strongly favors a non-spiking mean-membrane readout, while tuned output IM gives only a small improvement over naive spiking output.

## Repository Structure

- `train_temporal.py` - main training entry point for Randman and SHD.
- `eval_temporal.py` - checkpoint evaluation entry point.
- `models/temporal_snn.py` - LIF/RSNN classifiers, surrogate spike function, readout modes, and temporal IM losses.
- `data/randman.py` - deterministic random-manifold temporal spike dataset.
- `data/shd.py` - SHD HDF5 event loader and event-to-bin collation.
- `scripts/run_temporal_task.py` - cross-platform experiment task launcher.
- `scripts/smoke_temporal.py` - data-free smoke test.
- `summarize_ff_output_3way.py` - three-condition output/readout summary and bar plots.
- `summarize_lambda_sweep.py` - lambda sweep summaries and plots.
- `summarize_shd_results.py` - additional SHD multiseed summary utility.
- `notebooks/colab_run_experiments.ipynb` - Colab workflow used for GPU runs, summaries, plots, and final comparisons.
- `docs/technical_note.md` - method, results, limitations, and interpretation.
- `datasets/SHD/` - expected location for local SHD files.
- `im_snn_runs/summary/` - final CSV/JSON summaries and PNG figures used for reporting.

Large checkpoints and full run folders are not required for normal repository use. The final summary CSV/JSON/PNG files are enough to inspect and regenerate the reported figures.

## How To Run

### Environment

Use Python 3.10 or newer with a PyTorch build matching your machine. The Colab runs used Python 3.12, PyTorch 2.11.0+cu128, and an NVIDIA Tesla T4 GPU.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell activation on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Smoke Tests

These checks do not require SHD data:

```bash
python train_temporal.py --help
python eval_temporal.py --help
python scripts/smoke_temporal.py
python scripts/run_temporal_task.py ff_output_3way 0 --dry-run
```

A short CPU-only Randman run:

```bash
python train_temporal.py --dataset Randman --no_cuda --epochs 1 --max_train_batches 1 --max_eval_batches 1
```

### SHD Data

The SHD loader expects:

```text
datasets/SHD/shd_train.h5
datasets/SHD/shd_test.h5
```

If the downloaded files are compressed, decompress `shd_train.h5.gz` and `shd_test.h5.gz` first. The notebook also supports using SHD files stored in Google Drive.

### Main Command-Line Runs

Randman non-spiking membrane baseline:

```bash
python train_temporal.py --dataset Randman --output_mode nonspiking --readout max_membrane
```

Randman spiking output count:

```bash
python train_temporal.py --dataset Randman --output_mode spiking_count
```

Randman spiking output with output-layer rate IM:

```bash
python train_temporal.py --dataset Randman --output_mode spiking_count --use_im_loss --im_loss_type rate --im_include_output --im_hidden_weight 0.0
```

A single task-grid job:

```bash
python scripts/run_temporal_task.py ff_output_3way 0
```

A full 15-job three-condition grid is launched by running task IDs `0` to `14` for `ff_output_3way`. The notebook automates this loop.

### Lambda Sweep

Example Randman rate-IM sweep task:

```bash
IM_LAMBDAS=0,0.0001,0.0003,0.001,0.003,0.01 SEEDS=2020,42,123 LAMBDA_SWEEP_IM_LOSS_TYPE=rate python scripts/run_temporal_task.py ff_output_lambda_sweep 0
```

On PowerShell:

```powershell
$env:IM_LAMBDAS="0,0.0001,0.0003,0.001,0.003,0.01"
$env:SEEDS="2020,42,123"
$env:LAMBDA_SWEEP_IM_LOSS_TYPE="rate"
python scripts/run_temporal_task.py ff_output_lambda_sweep 0
```

### Summaries And Figures

Three-condition summary:

```bash
python summarize_ff_output_3way.py --dataset Randman --run_root im_snn_runs --output_dir im_snn_runs/summary
python summarize_ff_output_3way.py --dataset SHD --run_root im_snn_runs --output_dir im_snn_runs/summary
```

Lambda sweep summary:

```bash
python summarize_lambda_sweep.py --dataset Randman --loss_type rate --run_root im_snn_runs --output_dir im_snn_runs/summary
python summarize_lambda_sweep.py --dataset SHD --loss_type rate --run_root im_snn_runs --output_dir im_snn_runs/summary
```

The full Colab workflow is in `notebooks/colab_run_experiments.ipynb`. It was used to run the GPU experiments, collect summaries, and generate the final figures. Expected runtime for the full notebook is many hours on a Colab T4 GPU; for quick checks, run one-batch smoke tests or reduce the number of seeds.


## Author Contributions

Mu He and Fadi Ferjani jointly designed the study, reviewed the project plan, interpreted results, and prepared the final presentation material. Mu He focused on the temporal data pipeline, baseline/spiking readout comparisons, repository organization, and runnable experiment workflow. Fadi Ferjani focused on IM-loss experiment design, lambda sweeps, result summaries, plots, and scientific interpretation. Both authors contributed to debugging, documentation, and final result selection.

## Documentation Of LLM Usage

ChatGPT/Codex based on GPT-5 was used to assist with repository organization, code refactoring, notebook workflow design, documentation drafting, result interpretation, and presentation planning. The authors remain responsible for verifying the generated code, running the experiments, checking the saved outputs, and understanding the methods and conclusions.

## Citation

```bibtex
@inproceedings{
guo2022imloss,
title={{IM}-Loss: Information Maximization Loss for Spiking Neural Networks},
author={Yufei Guo and Yuanpei Chen and Liwen Zhang and Xiaode Liu and Yinglei Wang and Xuhui Huang and Zhe Ma},
booktitle={Advances in Neural Information Processing Systems},
year={2022}
}
```
