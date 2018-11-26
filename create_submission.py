import pandas as pd
import numpy as np
from util import util

raw_data = pd.read_csv('./dataset/test_V2.csv')
predictions = np.loadtxt('./test.out')
raw_data['winPlacePerc'] = predictions.tolist()
sub = raw_data[['Id', 'predictions']]
sub.to_csv('test_sub.csv', index=False)
exec(util.TEST_EMBEDDING)