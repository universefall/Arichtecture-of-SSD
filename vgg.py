import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.grad import Variable
import os

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 'C', 512, 512, 512, 
        'M', 512, 512, 512] #每个卷积层的通道数, C、M为最大池化，
"""
the shape of input and each convolution layer results:

300, 300,3 ->
Conv1_1 300,300,64 ->
Conv1_2 300,300,64 ->
Pooling1 150,150,64 ->

Conv2_1 150,150,128 ->
Conv2_2 150,150,128 ->
Pooling2 75,75, 128 ->

Conv3_1 75,75,256 ->
Conv3_2 75,75,256 ->
Conv3_3 38,38,256 ->
Pooling3 38,38,256 ->

Conv4_1 38,38,512 ->
Conv4_2 38,38,512 ->
conv4_3 38,38,512 ->

Conv5_1 19,19,512 ->
Conv5_2 19,19,512 ->
Conv5_3 19,19,512 ->

pooling5 19, 19, 152

conv6 19,19,1024
conv7 19,19,1024
"""

def vgg(i):
    layers = []
    # for 3 channels：
    in_channels = i
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            # 'C'表示ceil模式单数边
            layers += [nn.MaxPool2d(kernel_size=2,stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024,1024,kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers