'''
simple nerual network to regress the winPlacePerc

jiang wei
uvic
start: 2018.11.24
'''

import torch
import numpy as np
import torch.nn as nn

class PUBGSimpleRegressor(nn.Module):
    def __init__(self):
        super(PUBGSimpleRegressor, self).__init__()
        self.in_dim = 44
        self.out_dim = 1
        self.regressor = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.regressor(input)