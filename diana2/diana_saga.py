import numpy as np
import time
import os
import argparse

from mpi4py import MPI
from numpy.random import normal, uniform
from numpy.linalg import norm
from functions import logreg_loss
from quantization import quantize, decompress

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description='Run DIANA algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, help='Maximum number of iterations')
parser.add_argument('--max_t', action='store', dest='max_t', type=float, help='Time limit')
parser.add_argument('--alpha', action='store', dest='alpha', type=float, default=0, help='Rate of learning gradients')
parser.add_argument('--p_norm', action='store', dest='p_norm', type=int, default=2, help='Norm to be used in quantization. For infinity norm use zero value')
parser.add_argument('--batch', action='store', dest='batch', type=int, default=1, help='Minibatch size')
parser.add_argument('--block_size', action='store', dest='block_size', type=int, default=1, help='Block size for quantization')
parser.add_argument('--lr', action='store', dest='theta', type=float, default=0.1, help='Learning rate relative to smoothness')
parser.add_argument('-b', action='store_true', dest='big_reg', help='Whether to use 1/N regularization or 0.1/N')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='', help='Dataset name for saving logs')

args = parser.parse_args()

alpha = args.alpha
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
        
def stochastic_logreg_grad(w, X, y, i):
    return -y[i] * X[i] / (1 + np.exp(y[i] * (X[i] @ w)))

def sample_logreg_grads(w, z, X, y, l2, batch=1):
    n = len(y)
    idx = np.random.choice(n, size=batch, replace=False)
    ave_grad_w = l2 * w
    ave_grad_z = l2 * z
    for i in idx:
        ave_grad_w += stochastic_logreg_grad(w, X, y, i) / batch
        ave_grad_z += stochastic_logreg_grad(z, X, y, i) / batch
    return ave_grad_w, ave_grad_z
    
def generate_update(w, X, y, h_i, grads, grads_ave, l2=0, p_norm=2, batch=1, block_size=1):
    n, d = X.shape
    i = np.random.randint(n)
    stoch_grad_w = stochastic_logreg_grad(w, X, y, i) + l2 * w
    grad_estimate = grads_ave + stoch_grad_w - grads[i]
    Delta = grad_estimate - h_i
    block_norms, signs_quantized = quantize(Delta, p_norm, block_size)
    return block_norms, signs_quantized, i, stoch_grad_w

if rank == 0:
    X = np.load(data_path + 'X.npy')
    y = np.load(data_path + 'y.npy')
    N_X, d = X.shape
    n_workers_total = comm.reduce(0)
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
    grads = [np.zeros(d) for i in range(n_i)]
    grads_ave = np.zeros(d)
    while not np.isnan(w).any():
        comm.Bcast(w, root=0)
        if np.isnan(w).any():
            break

        block_norms, signs_quantized, i, stoch_grad_w = generate_update(w, X, y, h_i, grads, grads_ave, l2, p_norm, batch, block_size)
        grads_ave += (stoch_grad_w - grads[i]) / n_i
        grads[i] = np.copy(stoch_grad_w)

        h_i += alpha * decompress(block_norms, signs_quantized, d, p_norm, block_size)
        comm.Gather(block_norms, recvbuf=None, root=0)
        comm.Gather(signs_quantized, recvbuf=None, root=0)
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
    n_bytes = (2 * d) // 8 + (((2 * d)  % 8) != 0)
    
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

        Delta_i_hats = [decompress(deltas_norms[worker], deltas_signs[worker], d, p_norm, block_size) for worker in range(1, n_workers + 1)]
        Delta_hat = np.mean(Delta_i_hats, axis=0)
        assert len(Delta_hat) == d
        g_hat = h + Delta_hat
        h += alpha * Delta_hat
        w -= lr * g_hat

        ws.append(np.copy(w))
        information += n_workers * (2 * d + 64 * n_blocks) # A rough estimate
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
        
print("Rank %d is down" %rank)