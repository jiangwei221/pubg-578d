'''
useful util functions

jiang wei
uvic
start: 2018.11.24
'''

import torch

disable_embedding = False
TEST_EMBEDDING = '''
pass
''' if disable_embedding else '''
import IPython
IPython.embed()
assert(0)
'''

def to_torch(opt, nparray):
    if opt.use_cuda:
	    tensor = torch.from_numpy(nparray).cuda()
    else:
        tensor = torch.from_numpy(nparray)
    return torch.autograd.Variable(tensor,requires_grad=False)

def to_numpy(opt, cudavar):
    if opt.use_cuda:
	    return cudavar.data.cpu().numpy()
    else:
        return cudavar.data.numpy()

import time
def get_readable_cur_time():
    return time.ctime().replace(' ', '-')

import os.path as osp
def save_model(opt, model, it):
	torch.save(model.state_dict(), osp.join(opt.out, 'models_{0}_{1}.npy'.format(opt.model, it)))