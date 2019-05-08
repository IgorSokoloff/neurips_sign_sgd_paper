import numpy as np
import time
import os
import argparse

from mpi4py import MPI
from numpy.random import normal, uniform
from numpy.linalg import norm
from logreg_functions import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description='Run sign sgd algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, help='Maximum number of iterations')
parser.add_argument('--max_t', action='store', dest='max_t', type=float, help='Time limit')
parser.add_argument('--gamma_0', action='store', dest='gamma_0', type=float, default=0, help='Rate of learning gradients')
parser.add_argument('--p_norm', action='store', dest='p_norm', type=int, default=2,
                    help='Norm to be used in quantization. For infinity norm use zero value')
parser.add_argument('--batch', action='store', dest='batch', type=int, default=1, help='Minibatch size')
parser.add_argument('--block_size', action='store', dest='block_size', type=int, default=1,
                    help='Block size for quantization')
parser.add_argument('--lr', action='store', dest='theta', type=float, default=0.1,
                    help='Learning rate relative to smoothness')
parser.add_argument('-b', action='store_true', dest='big_reg', help='Whether to use 1/N regularization or 0.1/N')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='',
                    help='Dataset name for saving logs')

args = parser.parse_args()

gamma_0 = args.gamma_0
max_it = args.max_it
max_t = args.max_t
p_norm = args.p_norm
batch = args.batch
block_size = args.block_size
theta = args.theta
big_regularization = args.big_reg
dataset = args.dataset

if max_it is None:
    max_it = np.inf
if max_t is None:
    max_t = np.inf
if (max_it is np.inf) and (max_t is np.inf):
    raise ValueError('At least one stopping criterion must be specified')
if (gamma_0 < 0) or (gamma_0 >= 1):
    raise ValueError('gamma_0 values must lie within [0, 1) interval')

if dataset is None:
    dataset = "mushrooms"


######################
# This block varies between functions, options
loss_func = "log-reg"
#stepsize = "fix-step"

stepsize = "var-step"


def sample_sgrad(w, X, y, la, batch=1):
    #if log loss
    return sample_logreg_sgrad(w, X, y, la)

def update_stepsize(gamma_0, it):
    #if var-step
    return gamma_0/np.sqrt(it + 1)


#######################

def generate_update(w, X, y, la, gamma_0, it,batch=1):
    s_grad = sample_logreg_sgrad(w, X, y, la)
    gamma = update_stepsize(gamma_0, it)

    w = w - gamma * sign(s_grad)

    return w

def sign(arr):
    assert (isinstance(arr, (np.ndarray, np.generic) ))
    arr[arr==0] = 1
    return np.sign(arr)

n_workers = comm.Get_size() - 1
user_dir = os.path.expanduser('~/')

project_path = "/Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd"
logs_path = project_path + 'logs_{}/'.format(dataset)
data_path = project_path + "/data/"

data_info = np.load(data_path + 'data_info.npy')

N, L = data_info[:2]
Ls = data_info[2:]

experiment = 'sign_sgd_one_point_majority_{0}_{1}_{2}_{3}_{4}'.format(loss_func, stepsize, n_workers, gamma_0, batch)

if rank == 0:
    X = np.load(data_path + 'X.npy')
    y = np.load(data_path + 'y.npy')
    N_X, d = X.shape
    n_workers_total = comm.reduce(0)

    print ("total number of workers:", n_workers_total)

    assert n_workers_total == N, (N, n_workers_total)
    assert N_X == N

if rank > 0:
    X = np.load(data_path + 'Xs_' + str(rank - 1) + '.npy')
    y = np.load(data_path + 'ys_' + str(rank - 1) + '.npy')
    n_i, d = X.shape
    h_i = np.zeros(d)
    comm.reduce(n_i, root=0)
    w = np.zeros(shape=d)
    it = 0
    #grads = [np.zeros(d) for i in range(n_i)] ????

    while not np.isnan(w).any():
        comm.Bcast(w, root=0) # get_w from server
        if np.isnan(w).any():
            break

        w = generate_update(w, X, y, Ls[rank-1], gamma_0, it,batch=1)

        #grads[i] = np.copy(stoch_grad_w)

        it += 1

if rank == 0:
    w = np.zeros(d)
    ws = [np.copy(w)]
    information_sent = [0]
    ts = [0]
    its = [0]

    full_blocks = d // block_size
    has_tailing_block = ((d % block_size) != 0)
    n_blocks = full_blocks + has_tailing_block
    n_bytes = (2 * d) // 8 + (((2 * d) % 8) != 0)

    information = 0
    it = 0
    t_start = time.time()
    t = time.time() - t_start
    deltas_norms = np.empty(shape=[n_workers + 1, n_blocks])
    deltas_signs = np.empty(shape=[n_workers + 1, n_bytes], dtype='uint8')

    print(deltas_signs.shape)

    buff_norm = np.empty(shape=n_blocks)
    buff_signs = np.empty(shape=n_bytes, dtype='uint8')

    h = np.zeros(d)

    while (it < max_it) and (t < max_t):
        lr = theta / L_max

        assert len(w) == d
        comm.Bcast(w)
        comm.Gather(buff_norm, deltas_norms)
        comm.Gather(buff_signs, deltas_signs)

        Delta_i_hats = [decompress(deltas_norms[worker], deltas_signs[worker], d, p_norm, block_size) for worker in
                        range(1, n_workers + 1)]
        Delta_hat = np.mean(Delta_i_hats, axis=0)

        assert len(Delta_hat) == d
        g_hat = h + Delta_hat
        h += gamma_0 * Delta_hat
        w -= lr * g_hat

        ws.append(np.copy(w))
        information += n_workers * (2 * d + 64 * n_blocks)  # A rough estimate
        information_sent.append(information)
        t = time.time() - t_start
        ts.append(time.time() - t_start)
        its.append(it)
        it += 1

    print('Master: sending signal to all workers to stop.')
    # Interrupt all workers
    comm.Bcast(np.nan * np.zeros(d))

if rank == 0:
    print("There were done", len(ws), "iterations")
    step = len(ws) // 200 + 1
    loss = logreg_loss(ws[::step], X, y, l1=0, l2=l2)
    np.save(logs_path + 'loss' + '_' + experiment, np.array(loss))
    np.save(logs_path + 'time' + '_' + experiment, np.array(ts[::step]))
    np.save(logs_path + 'information' + '_' + experiment, np.array(information_sent[::step]))
    np.save(logs_path + 'iteration' + '_' + experiment, np.array(its[::step]))
    np.save(logs_path + 'iterates' + '_' + experiment, np.array(ws[::step]))
    print(loss)

print("Rank %d is down" % rank)