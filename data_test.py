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
    opt = options.set(training=True)
    pubg_data = data.PUBGDataset(opt)
    TRAIN_LEN = len(pubg_data)
    # exec(util.TEST_EMBEDDING)

    # pubg_data_small = Subset(pubg_data, range(0, 100))
    
    train_indices = range(0, TRAIN_LEN*9//10)
    valid_indices = range(TRAIN_LEN*9//10, TRAIN_LEN)
    pubg_train_loader = DataLoader(pubg_data, batch_size=32, num_workers=0, sampler=SubsetRandomSampler(train_indices))
    pubg_valid_loader = DataLoader(pubg_data, batch_size=32, num_workers=0, sampler=SubsetRandomSampler(valid_indices))
    
    #model
    pubg_reg = models.PUBGSimpleRegressor(opt)

    #optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, pubg_reg.parameters()), lr=opt.lr_classifier)

    my_trainer= trainer.Trainer(opt, pubg_reg, optimizer, pubg_train_loader, pubg_valid_loader)
    my_trainer.train()
    exec(util.TEST_EMBEDDING)
    

if __name__=='__main__':
    main()