# 0~3 right leg (up to down)
# 4~6 left leg (up to down)
# 7~10 body and head (down to up)
# 11~13 left hand (up to down)
# 14~16 right hand (up to down)

import os
import math
import shutil
import matplotlib
import numpy as np
import xgboost as xgb

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

_CONNECTION = [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
    [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
    [15, 16]]
_ACTION_PROPOSALS = [[0 + 1, 321], [342 + 1, 411], [673 + 1, 982], [1003 + 1, 1313], [1335 + 1, 1643], [1664 + 1, 1973], [1995 + 1, 2304], [2327 + 1, 2639]]
_INDEX_RANGE = range(1, 2638 + 1)
_COLOR = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
          (0, 255, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)]


# This part is data processing:
# 1. get all features
# 2. prepare for the format for xgboost

def consine_formula(x, y, z):
    '''
    summary:
    using consine formula to calculate angle xyz
    input:
    x, y, z means the points, format: [int, int, int]
    output:
    the angle xyz, from 0 to 180
    '''
    xy2 = math.pow(x[0] - y[0], 2) + math.pow(x[1] - y[1], 2) + math.pow(x[2] - y[2], 2)
    yz2 = math.pow(y[0] - z[0], 2) + math.pow(y[1] - z[1], 2) + math.pow(y[2] - z[2], 2)
    xz2 = math.pow(x[0] - z[0], 2) + math.pow(x[1] - z[1], 2) + math.pow(x[2] - z[2], 2)
    return math.acos((xy2 + yz2 - xz2) / (2 * math.sqrt(xy2 * yz2))) * 180 / math.pi


# consine_formula([1, 0, 0], [0, 0, 0], [0, 0, 1])
# consine_formula([1, 0, 0], [0, 0, 0], [-1, 0, 1])

def action_index2action_label(i):
    '''
    summary: since the label of action is pretty wrong, so such map is necessary
    0: 0, 6
    1: 1, 4
    2: 2, 5
    3: 3, 7
    input: the original action_label
    output: updated action_label
    '''
    _DICT = {0: [0, 6], 1: [1, 4], 2: [2, 5], 3: [3, 7]}
    for key in _DICT.keys():
        if i in _DICT[key]:
            return key


# for i in range(0, 8):
#     print(action_index2action_label(i))

def get_features_file():
    '''
    summary: get all the features from file
    feature1~6: relative pose of 15 - 9 and 16 - 9
    feature7: the angle of 14 15 16

    input: N/A
    output: two ndarray, which are required for xgboost
    '''
    all_X = []
    all_Y = []
    for index, action_type in enumerate(_ACTION_PROPOSALS):
        for file_index in range(action_type[0], action_type[1]):
            #             np.append(all_Y, action_index2action_label(action_index))
            all_Y.append(action_index2action_label(index))
            _POINTS = np.loadtxt('./3dpose/{}.txt'.format(file_index))
            all_X.append(
                [_POINTS[15][0] - _POINTS[9][0], _POINTS[15][1] - _POINTS[9][1], _POINTS[15][2] - _POINTS[9][2],
                 _POINTS[16][0] - _POINTS[9][0], _POINTS[16][1] - _POINTS[9][1], _POINTS[16][2] - _POINTS[9][2],
                 consine_formula(_POINTS[14].tolist(), _POINTS[15].tolist(), _POINTS[16].tolist())])
    all_X = np.array(all_X)
    all_Y = np.array(all_Y)

    return all_X, all_Y

# get_features_file()

# This part is to use xgboost
all_X, all_Y = get_features_file()
sz = all_X.shape

train_X = all_X[:int(sz[0] * 0.5), :]
train_Y = all_Y[:int(sz[0] * 0.5)]

test_X = all_X[int(sz[0] * 0.5):, :]
test_Y = all_Y[int(sz[0] * 0.5):]

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.3
param['max_depth'] = 6
# param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 4

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 5
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
print(pred[:100])
print(test_Y[:100])
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))

# This part is for understand this xgboost model
xgb.plot_importance(bst)
xgb.plot_tree(bst)
plt.show()


print(bst.predict(xg_test))
fig = plt.figure()
ax = fig.add_subplot('111')

# ax.plot(test_Y)
t1 = [i for i in range(0, 1120)]
t2 = bst.predict(xg_test).tolist()
ax.scatter([i for i in range(0, 1120)], bst.predict(xg_test).tolist(), c='red')
plt.show()