import numpy as np
import time
import os
import argparse

from mpi4py import MPI
from numpy.random import normal, uniform
from numpy.linalg import norm
from functions import logreg_loss

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description='Run QSVRG algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, help='Maximum number of iterations')
parser.add_argument('--max_t', action='store', dest='max_t', type=float, help='Time limit')
parser.add_argument('--batch', action='store', dest='batch', type=int, default=1, help='Minibatch size')
parser.add_argument('--l', action='store', dest='l', type=float, help='Loop length')
parser.add_argument('--lr', action='store', dest='theta', type=float, default=0.1, help='Learning rate relative to smoothness')
parser.add_argument('-b', action='store_true', dest='big_reg', help='Whether to use 1/N regularization or 0.1/N')

args = parser.parse_args()

max_it = args.max_it
max_t = args.max_t
batch = args.batch
block_size = args.block_size
l = args.l
theta = args.theta
big_regularization = args.big_reg
dataset = args.dataset

if max_it is None:
    max_it = np.inf
if max_t is None:
    max_t = np.inf
if (max_it is np.inf) and (max_t is np.inf):
    raise ValueError('At least one stopping criterion must be specified')
if (theta > 2):
    print('Warning: the method might diverge with lr bigger than 2')

n_workers = comm.Get_size() - 1
user_dir = os.path.expanduser('~/')
logs_path = user_dir + 'plots/diana/raw_data/'
data_path = user_dir + 'distr_opt/data_logreg/'

data_info = np.load(data_path + 'data_info.npy')
N, l2, L = data_info[:3]
big_regularization = (l2 == L / N)
Ls = data_info[3:]
L_max = np.max(Ls)

if l is None:
    l = 2 * N

experiment = 'qsvrg_{0}_{1}_{2}_{3}'.format(
    batch,
    block_size,
    'big' if big_regularization else 'small',
    theta
)
        
def stochastic_logreg_grad(w, X, y, i):
    return -y[i] * X[i] / (1 + np.exp(y[i] * (X[i] @ w)))

def sample_logreg_grads(w, z, X, y, l2, batch=1):
    n, d = X.shape
    assert len(w) == d
    assert len(z) == d
    assert len(y) == n
    idx = np.random.choice(n, size=batch, replace=False)
    ave_grad_w = l2 * w
    ave_grad_z = l2 * z
    for i in idx:
        ave_grad_w += stochastic_logreg_grad(w, X, y, i) / batch
        ave_grad_z += stochastic_logreg_grad(z, X, y, i) / batch
    assert len(ave_grad_w) == len(w)
    assert len(ave_grad_z) == len(z)
    return ave_grad_w, ave_grad_z

def logreg_fullgrad(w, X, y):
    n = len(y)
    full_grad = np.mean([stochastic_logreg_grad(w, X, y, i) for i in range(n)], axis=0)
    assert len(full_grad) == len(w)
    return full_grad

def generate_update(w, z, X, y, mu, l2=0, batch=1, block_size=1):
    stoch_grad_w, stoch_grad_z = sample_logreg_grads(w, z, X, y, l2)
    Delta = stoch_grad_w - stoch_grad_z
    return Delta

if rank == 0:
    X = np.load(data_path + 'X.npy')
    y = np.load(data_path + 'y.npy')
    N_X, d = X.shape
    assert N_X == N

if rank > 0:
    X = np.load(data_path + 'Xs_' + str(rank - 1) + '.npy')
    y = np.load(data_path + 'ys_' + str(rank - 1) + '.npy')
    n, d = X.shape
    h_i = np.zeros(d)
    w = np.zeros(shape=d)
    it = 0
    comm_time = 0
    full_grad_time = 0
    stoch_grad_time = 0
    while not np.isnan(w).any():
        t_start = time.time()
        comm.Bcast(w, root=0)
        comm_time += time.time() - t_start
        if np.isnan(w).any():
            break

        if ((it % l) == 0):
            z = np.copy(w)
            t_start = time.time()
            h_i = l2 * z + logreg_fullgrad(z, X, y)
            full_grad_time += time.time() - t_start
            comm.Reduce(None, h_i)
            t_start = time.time()
            comm_time += time.time() - t_start

        t_start = time.time()
        Delta = generate_update(w, z, X, y, l2, batch, block_size)
        stoch_grad_time += time.time() - t_start
        t_start = time.time()
        comm.Reduce(None, Delta)
        comm_time += time.time() - t_start
        it += 1
    print('Rank {0}: {1:.3f}, {2:.3f}, {3:.3f}'.format(rank, comm_time, full_grad_time, stoch_grad_time))

if rank == 0:
    w = np.zeros(d)
    ws = [np.copy(w)]
    information_sent = [0]
    ts = [0]
    its = [0]
    
    information = 0
    it = 0
    t_start = time.time()
    t = time.time() - t_start
    print(deltas_signs.shape)
    mu = np.zeros(d)
    Delta = np.zeros(d)
    while (it < max_it) and (t < max_t):
        lr = theta / L_max

        comm.Bcast(w)
        if ((it % l) == 0):
            comm.Reduce(mu)
            information += 2 * d * n_workers + 64 * n_workers * n_blocks
            assert len(mu) == d

        comm.Reduce(Delta)
        assert len(g_hat) == d
        w -= lr * (Delta + mu)

        ws.append(np.copy(w))
        information += 2 * d * n_workers + 64 * n_workers * n_blocks
        information_sent.append(information)
        t = time.time() - t_start
        ts.append(time.time() - t_start)
        its.append(it)
        it += 1

    print('Master: sending signal to all workers to stop.')
    for worker in range(1, n_workers + 1):
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
        
print("Rank %d is down" %rank)