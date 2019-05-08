import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
import os
import argparse

from numpy.random import normal, uniform
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix, load_svmlight_file
from numpy.linalg import norm

import sys

import itertools
from scipy.special import binom
from scipy.stats import ortho_group

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer as DV


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

parser = argparse.ArgumentParser(description='Generate data and provide information about it for workers and parameter server')
parser.add_argument('--n_workers', action='store', dest='n_workers', type=int, default=4, help='Number of workers that will be used')
parser.add_argument('--dataset', action='store', dest='dataset', default='real-sim', help='The name of the dataset')
parser.add_argument('-b', action='store_true', dest='big_reg', help='Whether to use 1/N regularization or 0.1/N')
parser.add_argument('--dimension', action='store', dest='dimension', type=int, default=300, help='Dimension for generating artifical data')
parser.add_argument('-l', action='store_true', dest='logistic', help='The problem is logistic regression')

args = parser.parse_args()
n_workers = args.n_workers
dataset = args.dataset
big_reg = args.big_reg
d = args.dimension
logistic = True

data_name = "mushrooms.txt"

user_dir = os.path.expanduser('~/')
SCRIPTS_PATH = '/Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/data/'
DATA_PATH = '/Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/data/'

def generate_data(d, min_cond=1e2, max_cond=1e4, diagonal=False):
    if diagonal:
        X = np.diag(uniform(low=10, high=1e3, size=d))
    else:
        ratio = np.inf
        while (ratio < min_cond) or (ratio > max_cond):
            X = 10 * make_spd_matrix(n_dim=d)
            vals, _ = np.linalg.eig(X)
            ratio = max(vals) / min(vals)
        print(ratio)
    y = 10 * uniform(low=-1, high=1, size=d)
    return X, y

def load_data(data_name):
    data = load_svmlight_file(DATA_PATH + data_name, zero_based=zero_based.get(dataset, 'auto'))
    return data[0], data[1]

Xs = []
ys = []

data, labels = load_svmlight_file(DATA_PATH + data_name)

enc_labels = labels.copy()
enc_labels[enc_labels==1] = -1
enc_labels[enc_labels==2] = 1
data_dense = data.todense()

train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(data_dense, enc_labels, test_size=0.2, random_state=0)

X = train_feature_matrix
y = train_labels
X_test = test_feature_matrix
y_test = test_labels

C = np.linspace(0.01, 1, 10)
param_grid_l2 = {'C': C, 'solver':['saga']}
cv = StratifiedKFold( n_splits=5, shuffle=True, random_state=0)
est_l2 = LogisticRegression(penalty='l2', max_iter=10000)

opt_l2 = GridSearchCV(est_l2, param_grid_l2, scoring = 'accuracy', cv=cv)
opt_l2.fit(train_feature_matrix, train_labels)

data_len = len(labels)

print('Number of data points:', data_len)

sep_idx = [0] + [(data_len * i) // n_workers for i in range(1, n_workers)] + [data_len]
# sep_idx = np.arange(0, n_workers * 100 + 1, 100)

la = 1 / opt_l2.best_params_['C']

data_info = [sep_idx[-1], la]

for i in range(n_workers):
    print('Creating chunk number', i + 1)
    start, end = sep_idx[i], sep_idx[i + 1]
    print(start, end)
    if logistic:
        Xs.append(X[start:end])
        ys.append(y[start:end])
        data_info.append(la)

# Remove old data
# os.system("bash -c 'rm {0}/Xs*'".format(SCRIPTS_PATH))
# os.system("bash -c 'rm {0}/ys*'".format(SCRIPTS_PATH))

# Save data for master
np.save(SCRIPTS_PATH + 'X', X)
np.save(SCRIPTS_PATH + 'y', y)
np.save(SCRIPTS_PATH + 'X_test', X_test)
np.save(SCRIPTS_PATH + 'y_test', y_test)

# Save data for workers
for worker in range(n_workers):
    np.save(SCRIPTS_PATH + 'Xs_' + str(worker), Xs[worker])
    np.save(SCRIPTS_PATH + 'ys_' + str(worker), ys[worker])
    np.save(SCRIPTS_PATH + 'data_info', data_info)