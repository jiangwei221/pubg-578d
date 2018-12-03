import torch
import data
import models
from options import options
from util import util
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch import optim
from trainer import trainer
# TRAIN_LEN = 4446966

def main():
    opt = options.set(training=False)
    pubg_data = data.PUBGInferDataset(opt)
    TRAIN_LEN = len(pubg_data)
    # exec(util.TEST_EMBEDDING)

    
    #model
    pubg_reg = models.PUBGSimpleRegressor(opt)
    state_dict = torch.load('test_weights.npy')
    pubg_reg.load_state_dict(state_dict)
    # exec(util.TEST_EMBEDDING)
    result = []
    for i, data_dict in enumerate(pubg_data):
        place = pubg_reg(data_dict['feat']).item()
        print(i, place)
        result += [place]
    
    import numpy as np 
    aa = np.array(result)
    np.savetxt('test.out', aa, delimiter=',', fmt='%f')
    # exec(util.TEST_EMBEDDING)
if __name__=='__main__':
    main()