import numpy as np
import time
import os
import argparse

from mpi4py import MPI
from numpy.random import normal, uniform
from numpy.linalg import norm


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from logreg_functions import logreg_loss, logreg_sgrad, sample_logreg_sgrad


def sign (a):
    a[a == 0] = 1
    return np.sign(a)

max_it = 1e+10
max_t = None


if max_it is None:
    max_it = np.inf
if max_t is None:
    max_t = np.inf
if (max_it is np.inf) and (max_t is np.inf):
    raise ValueError('At least one stopping criterion must be specified')
if (alpha < 0) or (alpha >= 1):
    raise ValueError('Alpha values must lie within [0, 1) interval')
if (theta > 2):
    print('Warning: the method might diverge with lr bigger than 2')
if (p_norm < 0):
    raise ValueError('p_norm must be non-negative')

n_workers = comm.Get_size() - 1
user_dir = os.path.expanduser('~/')
logs_path = user_dir + 'plots/diana/raw_data_{}/'.format(dataset)
data_path = user_dir + 'distr_opt/data_logreg/'

data_info = np.load(data_path + 'data_info.npy')
N, L = data_info[:2]
Ls = data_info[2:]
l2 = np.mean(Ls) / N if big_regularization else 0.1 * np.mean(Ls) / N
L_max = np.max(Ls)

experiment = 'diana_saga_{0}_{1}_{2}_{3}_{4}_{5}'.format(
    p_norm,
    alpha,
    batch,
    block_size,
    'big' if big_regularization else 'small',
    theta
)

if p_norm == 0:
    p_norm = np.inf