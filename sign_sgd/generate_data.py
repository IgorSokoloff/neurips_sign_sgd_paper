import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
import os
import argparse

from numpy.random import normal, uniform
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix, load_svmlight_file
from numpy.linalg import norm

from logreg_functions import *
from sigmoid_functions import *

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
parser.add_argument('--n_workers', action='store', dest='n_workers', type=int, default=3, help='Number of workers that will be used')
parser.add_argument('--dataset', action='store', dest='dataset', default='mushrooms', help='The name of the dataset')
parser.add_argument('-b', action='store_true', dest='big_reg', help='Whether to use 1/N regularization or 0.1/N')
parser.add_argument('--dimension', action='store', dest='dimension', type=int, default=300, help='Dimension for generating artifical data')
parser.add_argument('-l', action='store_true', dest='logistic', help='The problem is logistic regression')

args = parser.parse_args()
n_workers = args.n_workers
dataset = args.dataset
big_reg = args.big_reg
d = args.dimension

#loss_func = args.loss_func

loss_func_ar = ["log-reg", "sigmoid"]
data_name = dataset + ".txt"

user_dir = os.path.expanduser('~/')
SCRIPTS_PATH = '/Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/data/'
DATA_PATH = '/Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/data/'


project_path = "/Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/"
data_path = project_path + "data_{0}_{1}/".format(dataset, n_workers)

if not os.path.exists(data_path):
    os.mkdir(data_path)

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
data_dense = data.todense()

if not np.array_equal( np.unique(labels), np.array([-1,1],dtype='float')):
    enc_labels = labels.copy()
    enc_labels[enc_labels==1] = -1
    enc_labels[enc_labels==2] = 1

train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(data_dense, enc_labels, test_size=0.2, random_state=0)

X = np.array(train_feature_matrix)
y = np.array(train_labels)
X_test = np.array(test_feature_matrix)
y_test = np.array(test_labels)

"""
C = np.linspace(0.01, 1, 10)
param_grid_l2 = {'C': C, 'solver':['lbfgs']}
cv = StratifiedKFold( n_splits=5, shuffle=True, random_state=0)
est_l2 = LogisticRegression(penalty='l2', max_iter=1000000)

opt_l2 = GridSearchCV(est_l2, param_grid_l2, scoring = 'accuracy', cv=cv)
opt_l2.fit(train_feature_matrix, train_labels)

clf = LogisticRegression(penalty='l2', max_iter=1000000, C=opt_l2.best_params_['C'], solver='lbfgs')

clf.fit(train_feature_matrix, train_labels)
"""


X = np.array(train_feature_matrix)
y = np.array(train_labels)

data_len = len(labels)
train_data_len = X.shape[0]

#la = np.mean(np.diag(X.T @ X))

d = X.shape[1]
la = 1
w0 = np.random.normal(loc=0.0, scale=5.0, size=d)

"""
def f(w):
    return func(w, X, y, la, loss_func)
def grad(w):
    return grad(w, X, y, la, loss_func)
"""
for loss_func in loss_func_ar:
    if loss_func == "log-reg":
        f =    lambda w: logreg_loss(w, X, y, la)
        grad = lambda w: logreg_grad(w, X, y, la)
        result = minimize(fun=f, x0=w0, jac=grad, method="L-BFGS-B", options={"maxiter": 10000})

    if loss_func == "sigmoid":
        f =    lambda w: reg_bin_clf_loss(w, X, y)
        grad = lambda w: reg_bin_clf_grad(w, X, y)
        result = minimize (fun=f, x0=w0, jac=grad, method="Powell",options={"maxiter":10000})

    np.save(data_path + "{0}_clf_coef".format(loss_func), result.x)
    np.save(data_path + "{0}_f_min".format(loss_func), result.fun)



print('Number of data points:', data_len)

sep_idx = [0] + [(train_data_len * i) // n_workers for i in range(1, n_workers)] + [train_data_len]
# sep_idx = np.arange(0, n_workers * 100 + 1, 100)

data_info = [sep_idx[-1], la]

for i in range(n_workers):
    print('Creating chunk number', i + 1)
    start, end = sep_idx[i], sep_idx[i + 1]
    print(start, end)
    Xs.append(X[start:end])
    ys.append(y[start:end])
    data_info.append(la)

# Remove old data
# os.system("bash -c 'rm {0}/Xs*'".format(data_path))
# os.system("bash -c 'rm {0}/ys*'".format(data_path))

# Save data for master
np.save(data_path + 'X', X)
np.save(data_path + 'y', y)
np.save(data_path + 'X_test', X_test)
np.save(data_path + 'y_test', y_test)


# Save data for workers
for worker in range(n_workers):
    np.save(data_path + 'Xs_' + str(worker), Xs[worker])
    np.save(data_path + 'ys_' + str(worker), ys[worker])
    np.save(data_path + 'data_info', data_info)

