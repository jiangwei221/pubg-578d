'''
base network

jiang wei
uvic
start: 2018.11.24
'''

import torch
import numpy as np
import torch.nn as nn
from util import util

class PUBGBase(nn.Module):
    def __init__(self, opt):
        super(PUBGBase, self).__init__()
        self.in_dim = opt.in_dim
        self.out_dim = opt.out_dim
        if opt.model=='reg':
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, 128), nn.LeakyReLU(),
                nn.Linear(128, 128), nn.LeakyReLU(),
                nn.Linear(128, 128), nn.LeakyReLU(),
                nn.Linear(128, 128), nn.Tanh(),
                nn.Linear(128, self.out_dim)
            )
        elif opt.model=='ae':
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, 64), nn.LeakyReLU(),
                nn.Linear(64, 128), nn.LeakyReLU(),
                nn.Linear(128, 64), nn.LeakyReLU(),
                # nn.Dropout(),
                nn.Linear(64, 10), nn.LeakyReLU(),
                # nn.Dropout(),
                nn.Linear(10, self.out_dim)
            )

    def forward(self, x):
        # exec(util.TEST_EMBEDDING)
        return self.net(x)