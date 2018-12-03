'''
trainer
modified from pytorch-fcn, https://github.com/wkentaro/pytorch-fcn

jiang wei
uvic
start: 2018.11.25
'''

import datetime
import math
import os
import os.path as osp
import shutil

import numpy as np
import pytz
import torch
import torch.nn.functional as F
import tqdm
from util import util

from tensorboardX import SummaryWriter

class Trainer(object):

    def __init__(self, opt, model, optimizer, train_loader, val_loader):
        self.opt = opt
        self.cuda = opt.use_cuda
        if self.cuda:
            self.model = model.cuda()
        else:
            self.model = model
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

        self.out = opt.out
        if not osp.exists(self.out):
            os.makedirs(self.out)
        
        self.epoch = 0
        self.iteration = 0
        self.interval_validate = opt.valid_iter
        self.max_iter = opt.max_iter
        self.best_loss = float('inf')
        self.writer = SummaryWriter()

    def validate(self):
        training = self.model.training
        self.model.eval()

        val_loss = 0
        for batch_idx, data_dict in tqdm.tqdm(
            enumerate(self.val_loader), total=len(self.val_loader),
            desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            feat = data_dict['feat']
            target = data_dict['target']

            #infer
            with torch.no_grad():
                score = self.model(feat)
            #get loss
            if self.opt.model=='reg':
                loss = F.mse_loss(score, target)
            elif self.opt.model=='ae':
                loss = F.mse_loss(score, target)
                # huge_index = torch.abs(target) > 10
                # small_index = torch.abs(target) < 10
                # loss_1 = F.mse_loss(score[small_index], target[small_index])
                # diff = (target[huge_index] - score[huge_index]) / target[huge_index]
                # # diff = (target - score) / target
                # loss_2 = F.mse_loss(diff, torch.zeros(diff.shape).cuda())
                # loss = loss_1+loss_2
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                exec(util.TEST_EMBEDDING)
                raise ValueError('loss is nan while validating')
            val_loss += loss_data

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [val_loss]
                log = map(str, log)
                f.write(','.join(log) + '\n')
        util.save_model(self.opt, self.model, self.iteration)
        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        for batch_idx, data_dict in tqdm.tqdm(
            enumerate(self.train_loader), total=len(self.train_loader),
            desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iteration = batch_idx + self.epoch * len(self.train_loader)
            self.iteration = iteration
            if self.iteration % self.interval_validate == 0:
                self.validate()
            assert(self.model.training)

            feat = data_dict['feat']
            target = data_dict['target']

            #zero grad
            self.optim.zero_grad()
            #forward pass
            score = self.model(feat)
            #get loss
            if self.opt.model=='reg':
                loss = F.mse_loss(score, target)
            elif self.opt.model=='ae':
                loss = F.mse_loss(score, target)
                # huge_index = torch.abs(target) > 10
                # small_index = torch.abs(target) < 10
                # loss_1 = F.mse_loss(score[small_index], target[small_index])
                # diff = (target[huge_index] - score[huge_index]) / target[huge_index]
                # # diff = (target - score) / target
                # loss_2 = F.mse_loss(diff, torch.zeros(diff.shape).cuda())
                # loss = loss_1+loss_2
            # exec(util.TEST_EMBEDDING)
            # loss /= len(feat)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                exec(util.TEST_EMBEDDING)
                raise ValueError('loss is nan while training')
            #backward pass
            loss.backward()
            #update weights
            self.optim.step()

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
