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

parser = argparse.ArgumentParser(description='Run TERNGRAD-ADAM algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, help='Maximum number of iterations')
parser.add_argument('--max_t', action='store', dest='max_t', type=float, help='Time limit')
parser.add_argument('--block_size', action='store', dest='block_size', type=int, default=1, help='Block size for quantization')
parser.add_argument('--lr', action='store', dest='lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--beta1', action='store', dest='beta1', type=float, default=0.9, help='Numerator momentum')
parser.add_argument('--beta2', action='store', dest='beta2', type=float, default=0.999, help='Denominator momentum')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='', help='Dataset name for saving logs')
parser.add_argument('-b', action='store_true', dest='big_reg', help='Whether to use 1/N regularization or 0.1/N')

args = parser.parse_args()

max_it = args.max_it
max_t = args.max_t
block_size = args.block_size
lr = args.lr
big_regularization = args.big_reg
dataset = args.dataset
beta1 = args.beta1
beta2 = args.beta2

eps_denominator = 1e-4

p_norm = np.inf

if max_it is None:
    max_it = np.inf
if max_t is None:
    max_t = np.inf
if (max_it is np.inf) and (max_t is np.inf):
    raise ValueError('At least one stopping criterion must be specified')

n = comm.Get_size() - 1
user_dir = os.path.expanduser('~/')
logs_path = user_dir + 'plots/diana/raw_data_{}/'.format(dataset)
data_path = user_dir + 'distr_opt/data_logreg/'

data_info = np.load(data_path + 'data_info.npy')
N, L = data_info[:2]
Ls = data_info[2:]
l2 = np.mean(Ls) / N if big_regularization else 0.1 * np.mean(Ls) / N
L_max = np.max(Ls)

experiment = 'adam_{0}_{1}_{2}_{3}_{4}'.format(
    block_size,
    beta1,
    beta2,
    'big' if big_regularization else 'small',
    lr
)

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

def sample_logreg_grad(w, X, y, l2):
    n = len(y)
    i = np.random.choice(n)
    grad = l2 * w
    grad += stochastic_logreg_grad(w, X, y, i)
    return grad
    
def generate_update(w, X, y, m, v, l2=0, block_size=1):
    stoch_grad = sample_logreg_grad(w, X, y, l2=l2)
    Delta = stoch_grad
    block_norms, signs_quantized = quantize(Delta, p_norm=p_norm, block_size=block_size)
    return block_norms, signs_quantized

if rank == 0:
    X = np.load(data_path + 'X.npy')
    y = np.load(data_path + 'y.npy')
    N_X, d = X.shape
    assert N_X == N

if rank > 0:
    X = np.load(data_path + 'Xs_' + str(rank - 1) + '.npy')
    y = np.load(data_path + 'ys_' + str(rank - 1) + '.npy')
    d = X.shape[1]
    w = np.zeros(shape=d)
    m = np.zeros(d)
    v = np.zeros(d)
    while not np.isnan(w).any():
        comm.Bcast(w, root=0)
        if np.isnan(w).any():
            break
        block_norms, signs_quantized = generate_update(w=w, X=X, y=y, m=m, v=v, l2=l2, block_size=block_size)

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
    buff_norm = np.empty(shape=n_blocks)
    buff_signs = np.empty(shape=n_bytes, dtype='uint8')
    h = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)
    while (it < max_it) and (t < max_t):
        comm.Bcast(w)
        comm.Gather(buff_norm, deltas_norms)
        comm.Gather(buff_signs, deltas_signs)
        
        Delta_hat = np.zeros(d)
        for worker in range(1, n + 1):
            Delta_i_hat = decompress(block_norms=deltas_norms[worker], signs_compressed=deltas_signs[worker], d=d, p_norm=p_norm, block_size=block_size)
            Delta_hat += Delta_i_hat / n
        m = beta1 * m + (1 - beta1) * Delta_hat
        v = beta2 * v + (1 - beta1) * Delta_hat ** 2
        step = m / (np.sqrt(v) + eps_denominator)
        w -= lr * (1 - beta2 ** (it + 1)) / (1 - beta1 ** (it + 1)) * step

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