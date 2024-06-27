import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import math
import os

'''GPU/CPU'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trail = 10

'''neuron'''
decay = 0.5
num_classes = 10
thresh = 0.8
alpha = 2

'''network'''
conv_kernal = [(2, 32, 2, 3, 7),
           (32, 64, 1, 1, 3),
           (64, 128, 1, 1, 3),
           ]
conv_feature_size = [[64, 64],
              [32, 32],
              [16, 16],
              [8, 8],
              ]

fc_feature_size = [512, num_classes]

'''learning strategies'''
learning_rate = 1e-3
init_batch_size = 100
num_epochs = 100
gamma = 0.95

'''saving model'''
name = 'mnistdvs_4_' + 'decay' + str(decay) + 'vth' + str(thresh) + '_' + str(init_batch_size)
savemodel_path = './savemodel/' + name
os.makedirs(savemodel_path, exist_ok=True)

'''dataset load'''
with open('./path_4.txt', 'r') as f:
    datapath = f.readlines()
datasetPath = './mnistdvs/'

class MyDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, Train=True):
        self.path = datasetPath
        self.samples = sampleFile
        self.Train = Train

    def __getitem__(self, index):
        samples = self.samples[index].split()
        dataPath = self.path+samples[0]
        classLabel = int(samples[1])
        data = np.load(dataPath)
        data = data.f.data
        data = torch.from_numpy(data)
        return data, classLabel

    def __len__(self):
        return  len(self.samples)

'''surrogate gradient'''
class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = (alpha / 2) / (1 + ((alpha * math.pi / 2) * (x)) ** 2) * grad_output
        return grad_x

'''membrane potential update'''
def mem_update(ops, x, mem, spike, BN=False):
    if BN:
        mem = mem * decay * (1. - spike) + BN(ops(x))
    else:
        mem = mem * decay * (1. - spike) + ops(x)
    spike = atan.apply(mem-thresh)
    return mem, spike

'''attention'''
class attention(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=4):
        super(attention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_planes, out_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, in_planes, bias=False),
            nn.Sigmoid()
        )
        self.avgconv = nn.Conv2d(out_planes, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def spatial(self, input):
        return self.sigmoid(self.avgconv(input))

    def channel(self, input):
        b, c, d, e = input.size()
        return self.fc(self.avgpool(input).view(b, c)).view(b, c, 1, 1)

    def mixed(self, input):
        return torch.mul(self.spatial(input), self.channel(input))
