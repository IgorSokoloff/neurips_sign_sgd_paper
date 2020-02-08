from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n = 10
if rank == 0:
    sendbuf = np.full([n], -2,dtype='int8')
if rank > 0:
    arr = np.array([-1,1]*int(n/2), dtype='int8')
    np.random.shuffle(arr)
    sendbuf = arr


print("rank: {0}; send to server: {1}".format(rank, sendbuf))
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, n], dtype='int8')

if rank > 0:
    comm.Gather(sendbuf, recvbuf,root=0)

if rank == 0:
    comm.Gather(sendbuf, recvbuf,root=0)

if rank == 0:
    print("server recieved:\n {0}".format(recvbuf))
