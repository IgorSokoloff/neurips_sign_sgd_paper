import numpy as np
import time
import os
import argparse

from mpi4py import MPI
from numpy.random import normal, uniform
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
from numpy.linalg import norm
from functions import logreg_loss

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description='Run DIANA algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, help='Maximum number of iterations')
parser.add_argument('--max_t', action='store', dest='max_t', type=float, help='Time limit')
parser.add_argument('--alpha', action='store', dest='alpha', type=float, default=0, help='Rate of learning gradients')
parser.add_argument('--p_norm', action='store', dest='p_norm', type=int, default=2, help='Norm to be used in quantization')
parser.add_argument('--batch', action='store', dest='batch', type=int, default=1, help='Minibatch size')
parser.add_argument('--block_size', action='store', dest='block_size', type=int, default=1, help='Block size for quantization')
parser.add_argument('--momentum', action='store', dest='momentum', type=float, default=0, help='Heavy-ball momentum')
parser.add_argument('--theta', action='store', dest='theta', type=float, default=0.1, help='Constant part of the stepsize denominator')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='', help='Dataset name for saving logs')
parser.add_argument('-b', action='store_true', dest='big_reg', help='Whether to use 1/N regularization or 0.1/N')

args = parser.parse_args()

alpha = args.alpha
max_it = args.max_it
max_t = args.max_t
p_norm = args.p_norm
batch = args.batch
block_size = args.block_size
momentum = args.momentum
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
    raise ValueError('Alpha values must lie with [0, 1) interval')
if (momentum < 0) or (momentum >= 1):
    raise ValueError('Momentum values must lie with [0, 1) interval')

n = comm.Get_size() - 1
user_dir = os.path.expanduser('~/')
logs_path = user_dir + 'plots/diana/raw_data_{}/'.format(dataset)
data_path = user_dir + 'distr_opt/data_logreg/'

data_info = np.load(data_path + 'data_info.npy')
N, L = data_info[:2]
Ls = data_info[2:]
l2 = np.mean(Ls) / N if big_regularization else 0.1 * np.mean(Ls) / N
L_max = np.max(Ls)

experiment = 'diana_sgd_{0}_{1}_{2}_{3}_{4}_{5}'.format(
    p_norm, 
    alpha,
    batch,
    block_size,
    'big' if big_regularization else 'small',
    momentum
)

if p_norm == 0:
    p_norm = np.inf

def block_gen(v, block_size):
    d = len(v)
    current_block = 0
    full_blocks = d // block_size
    has_tailing_block = ((d % block_size) != 0)
    n_blocks = full_blocks + has_tailing_block
    while current_block < n_blocks:
        yield v[current_block * block_size: min(d, (current_block + 1) * block_size)]
        current_block += 1
        
def stochastic_logreg_grad(w, X, y, i):
    return -y[i] * X[i] / (1 + np.exp(y[i] * (X[i] @ w)))

def sample_logreg_grad(w, X, y, l2, batch=1):
    n = len(y)
    idx = np.random.choice(n, size=batch, replace=False)
    ave_grad = l2 * w
    for i in idx:
        ave_grad += stochastic_logreg_grad(w, X, y, i) / batch
    return ave_grad
    
def quantize_single_block(v_block, p_norm=2):
    block_norm = np.linalg.norm(v_block, ord=p_norm)
    if (block_norm == 0):
        return v_block
    xi = np.random.uniform(size=len(v_block)) < (np.abs(v_block) / block_norm)
    return xi * np.sign(v_block)
        
def quantize(v, p_norm=2, block_size=1):
    block_gen_for_norms = block_gen(v, block_size)
    block_norms = np.array([np.linalg.norm(v_block, ord=p_norm) for v_block in block_gen_for_norms])
    block_gen_for_signs = block_gen(v, block_size)
    signs_quantized = np.concatenate([quantize_single_block(v_block, p_norm) for v_block in block_gen_for_signs])
    pos_neg = np.concatenate([signs_quantized > 0, signs_quantized < 0])
    return block_norms, np.packbits(pos_neg)
    
def generate_update(w, X, y, h_i, l2=0, p_norm=2, batch=1, block_size=1):
    stoch_grad = sample_logreg_grad(w, X, y, l2)
    Delta = stoch_grad - h_i
    block_norms, signs_quantized = quantize(Delta, p_norm, block_size)
    return block_norms, signs_quantized

def decompress(block_norms, signs_compressed, d, p_norm=2, block_size=1):
    decompressed = np.unpackbits(signs_compressed)[:2 * d]
    decompressed_positive = decompressed[:d]
    decompressed_negative = decompressed[d:]
    signs = decompressed_positive * 1. - decompressed_negative * 1.
    sign_blocks = block_gen(signs, block_size)
    decompressed = np.concatenate([block_norms[i_block] * sign_block for i_block, sign_block in enumerate(sign_blocks)])
    assert len(decompressed) == d
    return decompressed

if rank == 0:
    X = np.load(data_path + 'X.npy')
    y = np.load(data_path + 'y.npy')
    N_X, d = X.shape
    assert N_X == N

if rank > 0:
    X = np.load(data_path + 'Xs_' + str(rank - 1) + '.npy')
    y = np.load(data_path + 'ys_' + str(rank - 1) + '.npy')
    d = X.shape[1]
    h_i = np.zeros(d)
    w = np.zeros(shape=d)
    while not np.isnan(w).any():
        comm.Bcast(w, root=0)
        if np.isnan(w).any():
            break
        block_norms, signs_quantized = generate_update(w, X, y, h_i, l2, p_norm, batch, block_size)

        h_i += alpha * decompress(block_norms, signs_quantized, d, p_norm, block_size)
        comm.Gather(block_norms, recvbuf=None, root=0)
        comm.Gather(signs_quantized, recvbuf=None, root=0)

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
    omega = np.sqrt(block_size)
    
    information = 0
    it = 0
    t_start = time.time()
    t = time.time() - t_start
    deltas_norms = np.empty(shape=[n + 1, n_blocks])
    deltas_signs = np.empty(shape=[n + 1, n_bytes], dtype='uint8')
    print(deltas_signs.shape)
    buff_norm = np.empty(shape=n_blocks)
    buff_signs = np.empty(shape=n_bytes, dtype='uint8')
    h = np.zeros(d)
    v = np.zeros(d) # Gradient estimate
    while (it < max_it) and (t < max_t):
        lr = 2 / (theta * L + 10 * l2 * it)
        
        comm.Bcast(w)
        comm.Gather(buff_norm, deltas_norms)
        comm.Gather(buff_signs, deltas_signs)
        
        Delta_hat = np.zeros(d)
        for worker in range(1, n + 1):
            Delta_i_hat = decompress(deltas_norms[worker], deltas_signs[worker], d, p_norm, block_size)
            Delta_hat += Delta_i_hat / n
        g_hat = h + Delta_hat
        v = momentum * v + g_hat
        h += alpha * Delta_hat
        w -= lr * v

        ws.append(np.copy(w))
        information += 2 * d * n + 32 * n * n_blocks # A rough estimate
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