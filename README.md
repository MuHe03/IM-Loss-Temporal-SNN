# IM-Loss: Information Maximization Loss for Spiking Neural Networks

Official simplified implementation of IM-Loss.

## Introduction

The forward-passing spike quantization will cause information loss and accuracy degradation. To deal with this problem, the Information maximization loss (IM-Loss) that aims at maximizing the information flow in the SNN is proposed in the paper.

### Dataset

The dataset will be download automatically.

## Get Started

```
cd imloss
python main_train.py --spike --step 1 --distribution
```

## SHD Temporal Experiments

The Python training and evaluation entry points remain at the repository root:

```bash
python train_temporal.py --help
python eval_temporal.py --help
```

Training wrappers run one array task at a time:

```bash
scripts/train_temporal_multiseed_local.sh 0
scripts/train_ff_output_3way_local.sh 5
scripts/train_ff_output_threshold_im_local.sh 0
scripts/eval_temporal_local.sh runs/.../best.pth
```

Override the interpreter or device through environment variables when needed:

```bash
PYTHON=python CUDA_VISIBLE_DEVICES=0 scripts/train_ff_output_3way_local.sh 5
```

## Citation

```bash
@inproceedings{
guo2022imloss,
title={{IM}-Loss: Information Maximization Loss for Spiking Neural Networks},
author={Yufei Guo and Yuanpei Chen and Liwen Zhang and Xiaode Liu and Yinglei Wang and Xuhui Huang and Zhe Ma},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022}
}
```
