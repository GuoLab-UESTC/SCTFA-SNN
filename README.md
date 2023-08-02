# A Spatial-Channel-Temporal-Fused Attention for Spiking Neural Networks

This repository is an example the codes for the paper [A Spatial-Channel-Temporal-Fused Attention for Spiking Neural Networks](https://ieeexplore.ieee.org/abstract/document/10138927). 

## Requirement
- Python 3.6
- torch 1.4.0
- torchvision 0.5.0
- CUDA 10.0
- MNIST-DVS dataset

## Running Example
1. Running the timeslice.py to split MNIST-DVS dataset into 20 time windows with 25ms;
2. Train the SCTFA-SNN model by running main.py.


## Citation
If you find this repository useful for your research, please consider citing the paper

```
@article{cai2023spatial,
  author={Cai, Wuque and Sun, Hongze and Liu, Rui and Cui, Yan and Wang, Jun and Xia, Yang and Yao, Dezhong and Guo, Daqing},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={A Spatial–Channel–Temporal-Fused Attention for Spiking Neural Networks}, 
  year={in press},
  doi={10.1109/TNNLS.2023.3278265}}
```
