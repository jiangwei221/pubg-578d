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
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.raw_data = pd.read_csv(file_path)
        self.init_keys = INIT_KEYS
        self.processed_data = self.feature_normalize(self.raw_data)
        self.np_feat, self.np_target = self.covert_to_np(self.processed_data)
        exec(util.TEST_EMBEDDING)

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
        feat = pd_dataframe.drop(CAT_KEYS+OUTPUT_KEYS, axis=1).as_matrix().astype('float32')
        target = pd_dataframe[OUTPUT_KEYS].as_matrix().astype('float32')

        return feat, target