import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.grad import Variable
import os

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 'C', 512, 512, 512, 
        'M', 512, 512, 512] #每个卷积层的通道数
“”“
”“”
'''