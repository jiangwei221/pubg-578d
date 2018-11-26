'''
PUBG dataset dataloader

jiang wei
uvic
start: 2018.11.22
'''

'''
training data summary:
4446966 players
47965   matches
2026745 groups
in everage: 100 players per match, 2 players per group, 50 groups per match

test data summary:
1934174 players
?? matches
?? groups

entry 2744604 contains nan
'''

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from util import util

INIT_KEYS = ['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 'longestKill', 'matchDuration', 'matchType', 'maxPlace','numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints', 'winPlacePerc']
CAT_KEYS = ['Id', 'groupId', 'matchId', 'matchType']
OUTPUT_KEYS = ['winPlacePerc']

class PUBGDataset(Dataset):
    def __init__(self, opt, transform=None):
        self.model = opt.model
        self.file_path = opt.training_file_path
        print('reading csv file into pandas dataframe')
        self.raw_data = pd.read_csv(self.file_path)
        self.raw_data.drop(2744604, inplace=True)
        self.init_keys = INIT_KEYS
        print('pre-processing data')
        self.processed_data = self.feature_normalize(self.raw_data)
        print('coverting pandas dataframe to numpy array')
        self.np_feat, self.np_target = self.covert_to_np(self.processed_data)
        print('coverting numpy array to pytorch tensor')
        self.feat = util.to_torch(opt, self.np_feat)
        self.target = util.to_torch(opt, self.np_target)
        self.num_samples = self.np_feat.shape[0]
        # exec(util.TEST_EMBEDDING)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # exec(util.TEST_EMBEDDING)
        if self.model == 'reg':
            return {'feat':self.feat[idx], 'target':self.target[idx]}
        elif self.model == 'ae':
            return {'feat':self.feat[idx], 'target':self.feat[idx]}
        else:
            raise RuntimeError('unknown model type')

    def normalize_one_feat(self, pd_dataframe, feat_name):
        feat_norm_name = feat_name+'Norm'
        # pd_dataframe[feat_norm_name] = pd_dataframe[feat_name]*((100-pd_dataframe['playersJoined'])/100 + 1)

    def feature_normalize(self, pd_dataframe):
        #create playersJoined
        pd_dataframe['playersJoined'] = pd_dataframe.groupby('matchId')['matchId'].transform('count')

        #how many teamates
        pd_dataframe['numTeammates'] = pd_dataframe.groupby('groupId')['Id'].transform('nunique')

        #how many groups in the match
        pd_dataframe['numGroups'] = pd_dataframe.groupby('matchId')['groupId'].transform('nunique')

        #create normalized features based on playersJoined
        exec(util.TEST_EMBEDDING)
        self.normalize_one_feat(pd_dataframe, 'kills')
        self.normalize_one_feat(pd_dataframe, 'damageDealt')
        # pd_dataframe['killsNorm'] = pd_dataframe['kills']*((100-pd_dataframe['playersJoined'])/100 + 1)
        # pd_dataframe['damageDealtNorm'] = pd_dataframe['damageDealt']*((100-pd_dataframe['playersJoined'])/100 + 1)

        #normalize some big values
        pd_dataframe['headshotNorm'] = (pd_dataframe.headshotKills - pd_dataframe.headshotKills.min()) / (pd_dataframe.headshotKills.max() - pd_dataframe.headshotKills.min())*((100-pd_dataframe['playersJoined'])/100 + 1) 
        pd_dataframe['damageNorm'] = (pd_dataframe.damageDealt - pd_dataframe.damageDealt.min()) / (pd_dataframe.damageDealt.max() - pd_dataframe.damageDealt.min())*((100-pd_dataframe['playersJoined'])/100 + 1)
        pd_dataframe['killsNorm'] = (pd_dataframe.killStreaks - pd_dataframe.killStreaks.min()) / (pd_dataframe.killStreaks.max() - pd_dataframe.killStreaks.min())*((100-pd_dataframe['playersJoined'])/100 + 1)

        #one-hot encoding for matchType
        pd_dataframe = pd.concat([pd_dataframe, pd.get_dummies(pd_dataframe['matchType'], prefix='matchType')],axis=1)

        return pd_dataframe

    def covert_to_np(self, pd_dataframe):
        # exec(util.TEST_EMBEDDING)
        feat = pd_dataframe.drop(CAT_KEYS+OUTPUT_KEYS, axis=1).values.astype('float32')
        target = pd_dataframe[OUTPUT_KEYS].values.astype('float32')
        assert(feat.shape[0] == target.shape[0])
        return feat, target


class PUBGInferDataset(Dataset):
    def __init__(self, opt, transform=None):
        self.model = opt.model
        self.file_path = opt.testing_file_path
        print('reading csv file into pandas dataframe')
        self.raw_data = pd.read_csv(self.file_path)
        self.init_keys = INIT_KEYS
        print('pre-processing data')
        self.processed_data = self.feature_normalize(self.raw_data)
        print('coverting pandas dataframe to numpy array')
        self.np_feat = self.covert_to_np(self.processed_data)
        print('coverting numpy array to pytorch tensor')
        self.feat = util.to_torch(opt, self.np_feat)
        self.num_samples = self.np_feat.shape[0]
        # exec(util.TEST_EMBEDDING)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # exec(util.TEST_EMBEDDING)
        if self.model == 'reg':
            return {'feat':self.feat[idx]}
        elif self.model == 'ae':
            return {'feat':self.feat[idx]}
        else:
            raise RuntimeError('unknown model type')

    def feature_normalize(self, pd_dataframe):
        #create playersJoined
        pd_dataframe['playersJoined'] = pd_dataframe.groupby('matchId')['matchId'].transform('count')

        #how many teamates
        pd_dataframe['numTeammates'] = pd_dataframe.groupby('groupId')['Id'].transform('nunique')

        #how many groups in the match
        pd_dataframe['numGroups'] = pd_dataframe.groupby('matchId')['groupId'].transform('nunique')

        #create normalized features based on playersJoined
        pd_dataframe['killsNorm'] = pd_dataframe['kills']*((100-pd_dataframe['playersJoined'])/100 + 1)
        pd_dataframe['damageDealtNorm'] = pd_dataframe['damageDealt']*((100-pd_dataframe['playersJoined'])/100 + 1)

        #one-hot encoding for matchType
        pd_dataframe = pd.concat([pd_dataframe, pd.get_dummies(pd_dataframe['matchType'], prefix='matchType')],axis=1)

        return pd_dataframe

    def covert_to_np(self, pd_dataframe):
        # exec(util.TEST_EMBEDDING)
        feat = pd_dataframe.drop(CAT_KEYS, axis=1).values.astype('float32')
        return feat