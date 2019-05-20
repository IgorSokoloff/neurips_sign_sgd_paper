import numpy as np
import time
import sys
import os
import argparse

from mpi4py import MPI
from numpy.random import normal, uniform
from numpy.linalg import norm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

max_it = 1
max_t = 1000

def sign(arr):
    assert (isinstance(arr, (np.ndarray, np.generic) ))
    arr[arr==0] = 1
    arr = np.sign(arr)
    return arr.astype('int8')

def sign_bool(arr):
    assert (isinstance(arr, (np.ndarray, np.generic) ))
    arr[arr==0] = 1
    arr = np.sign(arr)
    arr[arr==-1] = 0
    return arr.astype(bool)

if rank == 0:

    n_workers_total = comm.reduce(0)
    d = 10
    w = np.zeros(d, dtype='int8')
    print ("rank: {0}; total number of workers: {1}".format (rank, n_workers_total))

if rank > 0:

    n_i, d = 1, 10
    h_i = np.zeros(d)
    comm.reduce(n_i, root=0)
    w = np.zeros(shape=d)
    print (w)
    it = 0

    while not np.isnan(w).any():
        comm.Bcast(w, root=0) # get_w from server
        if np.isnan(w).any():
            break
        print("rank: {0}; recieve from server: {1}".format(rank, w))
        w = sign (np.random.uniform(low=-5,high=5,size=d))
        print("rank: {0}; send to server: {1}".format(rank, w))

        comm.Gather(w, None, root=0)

        it += 1

if rank == 0:

    information_sent = [0]
    ts = [0]
    its = [0]

    information = 0
    it = 0
    t_start = time.time()
    t = time.time() - t_start

    n_bytes= d
    send_buff = w.astype('int8')
    #send_buff = np.full([n_bytes],-2, dtype='int8')

    recv_buff = np.empty(shape=[size, n_bytes], dtype='int8')

    h = np.zeros(d)

    print (send_buff.shape, recv_buff.shape)

    while (it < max_it) and (t < max_t):

        assert len(w) == d
        comm.Bcast(w)

        comm.Gather(send_buff, recv_buff, root=0)

        print ("server recieved", recv_buff)
        t = time.time() - t_start
        ts.append(time.time() - t_start)
        its.append(it)
        it += 1

    print('Master: sending signal to all workers to stop.')
    # Interrupt all workers
    comm.Bcast(np.nan * np.zeros(d))
