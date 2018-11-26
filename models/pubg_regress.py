'''
simple nerual network to regress the winPlacePerc

jiang wei
uvic
start: 2018.11.24
'''

import torch
import numpy as np
import torch.nn as nn
from models import PUBGBase

class PUBGSimpleRegressor(nn.Module):
    def __init__(self, opt):
        super(PUBGSimpleRegressor, self).__init__()
        self.in_dim = opt.in_dim
        self.out_dim = opt.out_dim
        self.regressor = PUBGBase(opt)

    def forward(self, x):
        return torch.sigmoid(self.regressor(x))