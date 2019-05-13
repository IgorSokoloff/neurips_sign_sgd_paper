import numpy as np
import time
import os
import argparse

from mpi4py import MPI
from numpy.random import normal, uniform
from numpy.linalg import norm
from logreg_functions import *
from sigmoid_functions import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(description='Run sign sgd algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, help='Maximum number of iterations')
parser.add_argument('--max_t', action='store', dest='max_t', type=float, help='Time limit')
parser.add_argument('--gamma_0', action='store', dest='gamma_0', type=float, default=0, help='Rate of learning gradients')

parser.add_argument('--batch', action='store', dest='batch', type=int, default=1, help='Minibatch size')

parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms',
                    help='Dataset name for saving logs')

parser.add_argument('--step_type', action='store', dest='step_type', type=str, default='var-step',
                    help='variable or fixed stepsize ')

parser.add_argument('--loss_func', action='store', dest='loss_func', type=str, default='log-reg',
                    help='loss function ')

parser.add_argument('--upd_option', action='store', dest='upd_option', type=str, default='one-point',
                    help='option 1 or option 2')

parser.add_argument('--sampling_option', action='store', dest='sampling_option', type=str, default='non-block',
                    help='block or nonblock')


args = parser.parse_args()

gamma_0 = args.gamma_0
max_it = args.max_it
max_t = args.max_t

batch = args.batch
dataset = args.dataset
step_type = args.step_type
loss_func = args.loss_func
upd_option = args.upd_option
sampling_option = args.sampling_option

relax_number = 5

#print("options", step_type, loss_func, upd_option, dataset)

if max_it is None:
    max_it = np.inf
if max_t is None:
    max_t = np.inf
if (max_it is np.inf) and (max_t is np.inf):
    raise ValueError('At least one stopping criterion must be specified')
#if (gamma_0 < 0) or (gamma_0 >= 1):
 #   raise ValueError('gamma_0 values must lie within [0, 1) interval')

if step_type not in ["fix-step", "var-step"]:
    raise ValueError('only two options for step_type')

if upd_option not in ["one-point", "two-point"]:
    raise ValueError('only two options for update')

if loss_func not in ["log-reg", "sigmoid"]:
    raise ValueError('only two loss now is availible')

if sampling_option not in ["block", "non-block"]:
    raise ValueError('only block or non-block are availible')

if dataset is None:
    dataset = "mushrooms"

#print("dataset: {0}".format(dataset))

def myrepr(x):
    return repr(round(x, 2)).replace('.',',') if isinstance(x, float) else repr(x)

def sign(arr):
    assert (isinstance(arr, (np.ndarray, np.generic) ))
    arr[arr==0] = 1
    arr = np.sign(arr)
    return arr.astype('int8')

def get_std(arr, n):
    assert (n >= 0)
    arr = np.array(arr)
    #assert (isinstance(arr, (np.ndarray, np.generic)))
    if arr.shape[0] >=n:
        return arr[-n:].std()
    else:
        return arr.std()

######################
# This block varies between functions, options

def sample_sgrad(w, X, y, la, batch=1):
    if loss_func == "log-reg":
        return sample_logreg_sgrad(w, X, y, la)
    elif loss_func == "sigmoid":
        return sample_reg_bin_clf_sgrad(w, X, y)
    else:
        raise ValueError ('wrong loss_func')


def update_stepsize(gamma_0, it, power_step):
    if step_type == "var-step":
        return gamma_0/np.sqrt(it + 1)
    elif step_type == "fix-step":
        return gamma_0*(0.9**power_step)
        #return gamma_0
    else:
        raise ValueError ('wrong step_type')

def func(w, X, y, la):
    assert ((y.shape[0] == X.shape[0]) & (w.shape[0] == X.shape[1]))
    assert (la >= 0)
    #print(type(loss_func))

    if loss_func == "log-reg":
        return logreg_loss(w, X, y, la)
    elif loss_func == "sigmoid":
        return reg_bin_clf_loss(w, X, y)
    else:
        raise ValueError ('wrong loss_func')


def generate_update(w, X, y, loss_ar, power_step,  s_grad, la, gamma_0, it, batch=1):
    loss_cur = loss_ar[-1]
    gamma = update_stepsize(gamma_0, it, power_step)
    w_new = w - gamma * sign(s_grad)
    loss_new = func(w_new, X, y, la=la)
    if step_type == "fix-step":
        if get_std(loss_ar, relax_number):
            power_step += 1
    if upd_option == "one-point":
        return w_new, power_step
    elif upd_option == "two-point":
        if loss_new <= loss_cur:
            return w_new, power_step
        else:
            return w, power_step
    else:
        raise ValueError('wrong upd_option')

#######################


n_workers = comm.Get_size() - 1
user_dir = os.path.expanduser('~/')

project_path = "/Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/"

experiment_name = "sign_sgd_majority"

experiment = '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(experiment_name, upd_option,loss_func, step_type, n_workers, myrepr(gamma_0), batch)

logs_path = project_path + "logs_{0}_{1}/".format(dataset, experiment)
data_path = project_path + "data_{0}_{1}/".format(dataset, n_workers)

if rank==0:
    if os.path.exists(logs_path) == False:
        os.makedirs(logs_path)

    if os.path.exists(data_path) == False:
        os.mkdir(data_path)

data_info = np.load(data_path + 'data_info.npy')

N, L = data_info[:2]
Ls = data_info[2:]

#print("Ls:{0}".format(Ls))

experiment_name = "sign_sgd_majority"

experiment = '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(experiment_name, upd_option,loss_func, step_type, n_workers, myrepr(gamma_0), batch)


#save data of the experiment to disk
info_str = [experiment_name, project_path, logs_path, dataset, loss_func, upd_option]

np.save(logs_path + 'info_number' + "_" + experiment, np.array([step_type, n_workers, gamma_0, batch]))

with open(logs_path + 'info_str' + "_" + experiment + '.txt', 'w') as filehandle:
    for listitem in info_str:
        filehandle.write('%s\n' % listitem)

if rank == 0:
    print (experiment)
    X = np.load(data_path + 'X.npy')
    y = np.load(data_path + 'y.npy')
    N_X, d = X.shape

    if not os.path.isfile(data_path + 'w_init_{0}.npy'.format(loss_func)):
        if loss_func == "log-reg":
            w = np.random.normal(loc=0.0, scale=1.0, size=d)
            np.save(data_path + 'w_init_{0}.npy'.format(loss_func), w)
        elif loss_func == "sigmoid":
            w = np.random.uniform(low=-10, high=10, size=d)
            np.save(data_path + 'w_init_{0}.npy'.format(loss_func), w)
        else:
            raise ValueError('only two loss now is availible')
    else:
        w = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))

    #print(lipschitz_constant)

    data_length_total = comm.reduce(0)
    #print("data_length_total: {0}; lambda: {1}", data_length_total, L)

    #assert data_length_total == N, (N, data_length_total)
    #assert N_X == N

if rank > 0:
    if sampling_option == "block":
        X = np.load(data_path + 'Xs_' + str(rank - 1) + '.npy')
        y = np.load(data_path + 'ys_' + str(rank - 1) + '.npy')
    elif sampling_option == "non-block":
        X = np.load(data_path + 'X.npy')
        y = np.load(data_path + 'y.npy')
    else:
        raise ValueError('only block or non-block are availible')
    n_i, d = X.shape

    comm.reduce(n_i, root=0)
    w = np.zeros(d)# here we will broadcast it from server
    it = 0
    #grads = [np.zeros(d) for i in range(n_i)] ????

    while (np.isnan(w).any() == False) and (it < max_it + 2):
        comm.Barrier()
        comm.Bcast(w, root=0) # get_w from server
        #print("rank: {0}; recieve from server: {1}".format(rank, w))
        if np.isnan(w).any() == True:
            break
        #print("rank: {0}; recieve from server: {1}".format(rank, w))

        s_grad_sign = sign(sample_sgrad(w, X, y, Ls[rank-1]))

        #print("rank: {0}; send to server: {1}".format(rank, s_grad_sign))
        comm.Gather(s_grad_sign, None, root=0)
        #grads[i] = np.copy(stoch_grad_w)

        it += 1

if rank == 0:
    s_grad_sign = np.zeros(shape=d, dtype='int8')

    ws = [np.copy(w)]
    loss = [func(ws[-1], X, y, la=L)]

    power_step = 0

    ts = [0]
    its = [0]

    n_bytes = d
    send_buff = s_grad_sign

    recv_buff = np.empty(shape=[size, n_bytes], dtype='int8')

    it = 0
    t_start = time.time()
    t = time.time() - t_start

    while (it < max_it) and (t < max_t):
        print ("it: {0} , loss: {1}, loss_std: {2}".format(it, loss[-1], get_std(loss, relax_number)))
        #print("it: {0}".format(it))
        assert len(w) == d
        comm.Bcast(w)
        comm.Barrier()
        comm.Gather(send_buff, recv_buff, root=0)

        #print("recieve from workers:\n {0}".format(recv_buff))

        s_grad_major_vote = sign(np.sum(recv_buff[1:,:], axis=0))

        w, power_step  = generate_update(w, X, y, loss, power_step, s_grad_major_vote, L, gamma_0, it,batch=1)
        #print ("new_w: {0}".format(w))
        #comm.Bcast(w)

        ws.append(np.copy(w))
        loss.append(func(ws[-1], X, y, la=L))

        t = time.time() - t_start
        ts.append(time.time() - t_start)
        it += 1
        its.append(it)


    print('Master: sending signal to all workers to stop.')
    # Interrupt all workers
    comm.Barrier()
    comm.Bcast(np.nan * np.zeros(d))


if rank == 0:
    print("There were done", len(ws), "iterations")
    np.save(logs_path + 'loss' + '_' + experiment, np.array(loss))
    np.save(logs_path + 'time' + '_' + experiment, np.array(ts))
    #np.save(logs_path + 'information' + '_' + experiment, np.array(information_sent[::step]))
    np.save(logs_path + 'iteration' + '_' + experiment, np.array(its))
    np.save(logs_path + 'epochs' + '_' + experiment, np.array(its)/data_length_total)
    np.save(logs_path + 'iterates' + '_' + experiment, np.array(ws))
    #print(loss)


#just for test
#if rank == 0:
#    print ("loss: ",np.load(logs_path + 'loss' + '_' + experiment + ".npy"))
#    print("time: ", np.load(logs_path + 'time' + '_' + experiment + ".npy"))
#    print("iteration: ", np.load(logs_path + 'iteration' + '_' + experiment + ".npy"))
#    print("iterates: ", np.load(logs_path + 'iterates' + '_' + experiment + ".npy"))


print("Rank %d is down" % rank)