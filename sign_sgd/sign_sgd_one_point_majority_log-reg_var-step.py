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
size = comm.Get_size()

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
#if (gamma_0 < 0) or (gamma_0 >= 1):
 #   raise ValueError('gamma_0 values must lie within [0, 1) interval')

if dataset is None:
    dataset = "mushrooms"

#print("dataset: {0}".format(dataset))

######################
# This block varies between functions, options
loss_func = "log-reg"
#stepsize = "fix-step"

stepsize = "var-step"

def sample_sgrad(w, X, y, la, batch=1):
    #if log-reg
    return sample_logreg_sgrad(w, X, y, la)

def update_stepsize(gamma_0, it):
    #if var-step
    return gamma_0/np.sqrt(it + 1)

def func(w, X, y, la):

    assert ((y.shape[0] == X.shape[0]) & (w.shape[0] == X.shape[1]))
    assert (la >= 0)
    #if log-reg
    return logreg_loss(w, X, y, la)

#######################

def sign(arr):
    assert (isinstance(arr, (np.ndarray, np.generic) ))
    arr[arr==0] = 1
    arr = np.sign(arr)
    return arr.astype('int8')

def generate_update(w, X, y, s_grad, la, gamma_0, it,batch=1):
    gamma = update_stepsize(gamma_0, it)

    return w - gamma * sign(s_grad)

n_workers = comm.Get_size() - 1
user_dir = os.path.expanduser('~/')

project_path = "/Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/"

logs_path = project_path + "logs_{0}/".format(dataset)
data_path = project_path + "data/"

if not os.path.exists(logs_path):
    os.makedirs(logs_path)

if not os.path.exists(data_path):
    os.makedirs(data_path)

data_info = np.load(data_path + 'data_info.npy')

N, L = data_info[:2]
Ls = data_info[2:]

#print("Ls:{0}".format(Ls))

experiment = 'sign_sgd_one_point_majority_{0}_{1}_{2}_{3}_{4}'.format(loss_func, stepsize, n_workers, gamma_0, batch)

if rank == 0:
    X = np.load(data_path + 'X.npy')
    y = np.load(data_path + 'y.npy')
    N_X, d = X.shape

    #print(lipschitz_constant)

    data_length_total = comm.reduce(0)
    #print("data_length_total: {0}; lambda: {1}", data_length_total, L)

    #assert data_length_total == N, (N, data_length_total)
    #assert N_X == N

if rank > 0:
    X = np.load(data_path + 'Xs_' + str(rank - 1) + '.npy')
    y = np.load(data_path + 'ys_' + str(rank - 1) + '.npy')
    n_i, d = X.shape

    comm.reduce(n_i, root=0)
    w = np.zeros(d)# here we will broadcast it from server
    it = 0
    #grads = [np.zeros(d) for i in range(n_i)] ????

    while not np.isnan(w).any():
        comm.Bcast(w, root=0) # get_w from server
        if np.isnan(w).any():
            break
        #print("rank: {0}; recieve from server: {1}".format(rank, w))

        s_grad_sign = sign(sample_sgrad(w, X, y, Ls[rank-1]))

        #print("rank: {0}; send to server: {1}".format(rank, s_grad_sign))
        comm.Gather(s_grad_sign, None, root=0)
        #grads[i] = np.copy(stoch_grad_w)

        it += 1

if rank == 0:
    w = np.zeros(d)
    s_grad_sign = np.zeros(shape=d, dtype='int8')

    ws = [np.copy(w)]
    information_sent = [0]
    ts = [0]
    its = [0]

    n_bytes = d
    send_buff = s_grad_sign

    recv_buff = np.empty(shape=[size, n_bytes], dtype='int8')

    it = 0
    t_start = time.time()
    t = time.time() - t_start

    while (it < max_it) and (t < max_t):
        print ("it: {0} , loss: {1}".format(it, func(w, X, y, la=L)))
        assert len(w) == d
        comm.Bcast(w)

        comm.Gather(send_buff, recv_buff, root=0)

        #print("recieve from workers:\n {0}".format(recv_buff))

        s_grad_major_vote = sign(np.sum(recv_buff, axis=0))

        w  = generate_update(w, X, y, s_grad_major_vote, L, gamma_0, it,batch=1)
        #print ("new_w: {0}".format(w))
        comm.Bcast(w)

        ws.append(np.copy(w))

        t = time.time() - t_start
        ts.append(time.time() - t_start)
        it += 1
        its.append(it)


    print('Master: sending signal to all workers to stop.')
    # Interrupt all workers
    comm.Bcast(np.nan * np.zeros(d))

if rank == 0:
    print("There were done", len(ws), "iterations")
    loss = np.array([func(ws[i], X, y, la=L) for i in range(len(ws)) ])
    np.save(logs_path + 'loss' + '_' + experiment, np.array(loss))
    np.save(logs_path + 'time' + '_' + experiment, np.array(ts))
    #np.save(logs_path + 'information' + '_' + experiment, np.array(information_sent[::step]))
    np.save(logs_path + 'iteration' + '_' + experiment, np.array(its))
    np.save(logs_path + 'iteration' + '_' + experiment, np.array(its/data_length_total))
    np.save(logs_path + 'iterates' + '_' + experiment, np.array(ws))
    print(loss)

'''
#just for test
if rank == 0:
    print ("loss: ",np.load(logs_path + 'loss' + '_' + experiment + ".npy"))
    print("time: ", np.load(logs_path + 'time' + '_' + experiment + ".npy"))
    print("iteration: ", np.load(logs_path + 'iteration' + '_' + experiment + ".npy"))
    print("iterates: ", np.load(logs_path + 'iterates' + '_' + experiment + ".npy"))
'''

print("Rank %d is down" % rank)