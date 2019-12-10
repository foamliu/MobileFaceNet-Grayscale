# MobileFaceNets

PyTorch implementation of MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices.
[paper](https://arxiv.org/abs/1804.07573). 

## Features

1. Black-and-white photos for training/validation.
2. Retinaface & similarity transform for face alignment.
3. Lightweight: Params size (MB): 0.95, FLOPs size (GB): 0.24.

## Performance

|Accuracy|LFW|Download|
|---|---|---|
|paper|99.55%||
|ours|99.45%|[Link](https://github.com/foamliu/MobileFaceNet-v2/releases/download/v1.0/mobilefacenet_grayscale_scripted.pt)|

## Dataset
### Introduction

Refined MS-Celeb-1M dataset for training, 5,179,510 faces over 93,431 identities. 
LFW datasets for testing.

## Dependencies
- Python 3.6.8
- PyTorch 1.3.0

## Usage

### Data preprocess
Extract images:
```bash
$ python extract.py
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```



