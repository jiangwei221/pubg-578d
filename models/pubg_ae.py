'''
simple auto encoder for pugb data

jiang wei
uvic
start: 2018.11.24
'''

import torch
import numpy as np
import torch.nn as nn
from models import PUBGBase

class PUBGSimpleAE(nn.Module):
    def __init__(self, opt):
        super(PUBGSimpleAE, self).__init__()
        self.in_dim = opt.in_dim
        self.out_dim = opt.out_dim
        self.ae = PUBGBase(opt)

    def forward(self, x):
        return self.ae(x)