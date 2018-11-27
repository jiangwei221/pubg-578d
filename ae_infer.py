import torch
import data
import models
from options import options
from util import util
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch import optim
from trainer import trainer
import torch.nn.functional as F
# TRAIN_LEN = 4446966

def main():
    opt = options.set(training=True)
    pubg_data = data.PUBGDataset(opt)
    TRAIN_LEN = len(pubg_data)
    # exec(util.TEST_EMBEDDING)

    
    #model
    pubg_ae = models.PUBGSimpleAE(opt)
    state_dict = torch.load('ae_weights.npy')
    pubg_ae.load_state_dict(state_dict)

    result = []
    # for i, data_dict in enumerate(pubg_data):
        # exec(util.TEST_EMBEDDING)
    for i in range(800000, 1000000):
        data_dict = pubg_data[i]
        target = data_dict['feat']
        score = pubg_ae(target)
        huge_index = torch.abs(target) > 10
        small_index = torch.abs(target) < 10
        loss_1 = F.mse_loss(score[small_index], target[small_index])*1000
        diff = (target[huge_index] - score[huge_index]) / target[huge_index]
        # diff = (target - score) / target
        loss_2 = F.mse_loss(diff, torch.zeros(diff.shape))*1000
        loss = loss_1+loss_2
        print(i, loss.data.item())
        if loss>1000:
            result += [[i, loss.data.item()]]
    
    import numpy as np 
    aa = np.array(result)
    np.savetxt('test.out', aa, delimiter=',', fmt='%f')
    exec(util.TEST_EMBEDDING)
if __name__=='__main__':
    main()