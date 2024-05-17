# (To Be Updated)Towards Universal X-ray Security Inspection: A Benchmark and Stereoscopic-Aware Oriented Prohibited Items Detection Framework

## Introduction

This repository provides the code for the paper: Towards Universal X-ray Security Inspection: A Benchmark and Stereoscopic-Aware Oriented Prohibited Items Detection Framework


## Installation
Our code is implemented based on MMRotate, which depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create -n open-mmlab python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
```

## Training and Testing

```shell
sh scripts/train_ours.sh
```

