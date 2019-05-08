import numpy as np
import time
import os
import argparse

from mpi4py import MPI
from numpy.random import normal, uniform
from numpy.linalg import norm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:

    n_workers_total = comm.reduce(0)
    d = 10
    w = np.zeros(d)
    print ("rank: {0}; total number of workers: {1}".format (rank, n_workers_total))

if rank > 0:

    n_i, d = 1000, 10
    h_i = np.zeros(d)
    comm.reduce(n_i, root=0)
    w = np.zeros(shape=d)
    it = 0

    while not np.isnan(w).any():
        comm.Bcast(w, root=0) # get_w from server
        if np.isnan(w).any():
            break
        print("rank: {0}; recieve from server: {1}".format(rank, w))
        w = w - np.random.uniform(low=-5,high=5,size=d)
        it += 1

if rank == 0:
    w = np.zeros(d)

    information_sent = [0]
    ts = [0]
    its = [0]

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
