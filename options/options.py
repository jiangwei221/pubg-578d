'''
traning/test options

jiang wei
uvic
start: 2018.11.22
'''

import argparse

def set(training):
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", type=bool, default=True, help="normaliza the data based on playersJoined feature")
    if training:
        parser.add_argument("--lrC", type=float, default=1e-2, help="learning rate")
        parser.add_argument("--batchSize", type=int, default=32, help="batch size for training")
    else:
        parser.add_argument("--batchSize", type=int, default=1, help="batch size for evaluation")
    
    opt = parser.parse_args()

    return opt