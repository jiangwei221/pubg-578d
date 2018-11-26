'''
traning/test options

jiang wei
uvic
start: 2018.11.22
'''

import argparse
import torch
from util import util

def set(training):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['reg','ae'], help='type of network')
    parser.add_argument('--use_cuda', action='store_true', help='input dimension for network')
    if training:
        parser.add_argument('--training_file_path', type=str, default='./dataset/train_V2.csv', help='training file location')
        parser.add_argument('--lr_classifier', type=float, default=1e-2, help='classifier learning rate')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--max_iter', type=int, default=200000, help='total training iterations')
        parser.add_argument('--valid_iter', type=int, default=1000, help='iterval of validation')
        parser.add_argument('--out', type=str, default='./out/out-'+util.get_readable_cur_time(), help='training log')
    else:
        parser.add_argument('--testing_file_path', type=str, default='./dataset/test_V2.csv', help='training file location')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
    
    opt = parser.parse_args()
    opt.in_dim = 44
    opt.out_dim = opt.in_dim if opt.model=='ae' else \
                    1 if opt.model=='reg' else None

    if training:
        opt.training = True
    else:
        opt.training = False
    
    if opt.use_cuda:
        assert(torch.cuda.is_available())

    return opt