import numpy as np
import os
import itertools

from algorithms import svrg


def get_solution(X, y, data_path, dataset='mushrooms', big_regularization=True, S=None, eta=None):
    if S is None:
        S = 50 if big_regularization else 500
    if eta is None:
        eta = 0.1 / L
    n, d = X.shape
    regularization = 'big' if big_regularization else 'small'
    data_info = np.load(data_path + 'data_info.npy')
    N, L = data_info[:2]
    Ls = data_info[2:]
    l2 = np.mean(Ls) / N * (1 if big_regularization else 1e-1)
    solution_path = './solutions/w_star_{0}_{1}.npy'.format(dataset, regularization)
    if os.path.isfile(solution_path):
        print('Loading the solution from file')
        print(solution_path)
        w_star = np.load(solution_path)
        ws_svrg = None
    else:
        print('Computing the solution using SVRG')
        ws_svrg, _, _ = svrg(np.zeros(d), X, y, eta=eta, mu=l2, S=S, M=None, max_t=np.inf)
        w_star = ws_svrg[-1]
        np.save(solution_path, w_star)
    return w_star, ws_svrg


def run_diana_svrg(n_workers, code_path, max_it, max_t, alphas, p_norms, batches, block_sizes, ls, lrs,
                   big_regularization, dataset):
    for params in itertools.product(alphas, p_norms, batches, block_sizes, ls, lrs):
        alpha, p_norm, batch, block_size, l, lr = params
        os.system(
            "bash -c 'mpiexec -n {0} python {1}diana_svrg.py --max_it {2} --max_t {3} --alpha {4} --p_norm {5} --batch {6} --block_size {7} --l {8} --lr {9} {10} --dataset {11}'".format(
                n_workers + 1,
                code_path,
                max_it,
                max_t,
                alpha,
                p_norm,
                batch,
                block_size,
                l,
                lr,
                '-b' if big_regularization else '',
                dataset
            ))
    print('#', end='')


def run_diana_saga(n_workers, code_path, max_it, max_t, alphas, p_norms, batches, block_sizes, ls, lrs,
                   big_regularization, dataset):
    for params in itertools.product(alphas, p_norms, batches, block_sizes, ls, lrs):
        alpha, p_norm, batch, block_size, l, lr = params
        os.system(
            "bash -c 'mpiexec -n {0} python {1}diana_saga.py --max_it {2} --max_t {3} --alpha {4} --p_norm {5} --batch {6} --block_size {7} --lr {8} {9} --dataset {10}'".format(
                n_workers + 1,
                code_path,
                max_it,
                max_t,
                alpha,
                p_norm,
                batch,
                block_size,
                lr,
                '-b' if big_regularization else '',
                dataset
            ))


def run_diana_lsvrg(n_workers, code_path, max_it, max_t, alphas, p_norms, batches, block_sizes, ls, lrs,
                    big_regularization, dataset):
    for params in itertools.product(alphas, p_norms, batches, block_sizes, ls, lrs):
        alpha, p_norm, batch, block_size, l, lr = params
        os.system(
            "bash -c 'mpiexec -n {0} python {1}diana_lsvrg.py --max_it {2} --max_t {3} --alpha {4} --p_norm {5} --batch {6} --block_size {7} --lr {8} {9} --dataset {10}'".format(
                n_workers + 1,
                code_path,
                max_it,
                max_t,
                alpha,
                p_norm,
                batch,
                block_size,
                lr,
                '-b' if big_regularization else '',
                dataset
            ))


def run_qsvrg(n_workers, code_path, max_it, max_t, batches, block_sizes, ls, lrs, big_regularization, dataset):
    for params in itertools.product(batches, block_sizes, ls, lrs):
        batch, block_size, l, lr = params
        os.system(
            "bash -c 'mpiexec -n {0} python {1}qsvrg.py --max_it {2} --max_t {3} --batch {4} --block_size {5} --l {6} --lr {7} {8} --dataset {9}'".format(
                n_workers + 1,
                code_path,
                max_it,
                max_t,
                batch,
                block_size,
                l,
                lr,
                '-b' if big_regularization else '',
                dataset
            ))


def run_diana_sgd(n_workers, code_path, max_it, max_t, alphas, p_norms, batches,
                  block_sizes, thetas, momentums, big_regularization, dataset):
    for params in itertools.product(alphas, p_norms, batches, block_sizes, thetas, momentums):
        alpha, p_norm, batch, block_size, theta, momentum = params
        os.system(
            "bash -c 'mpiexec -n {0} python {1}diana_sgd.py --max_it {2} --max_t {3} --alpha {4} --p_norm {5} --batch {6} --block_size {7} --theta {8} --momentum {9} --dataset {10} {11}'".format(
                n_workers + 1,
                code_path,
                max_it,
                max_t,
                alpha,
                p_norm,
                batch,
                block_size,
                theta,
                momentum,
                dataset,
                '-b' if big_regularization else ''
            ))


def run_adam(n_workers, code_path, max_it, max_t, block_sizes, lrs, beta1s,
             beta2s, big_regularization, dataset):
    for params in itertools.product(block_sizes, lrs, beta1s, beta2s):
        block_size, lr, beta1, beta2 = params
        os.system(
            "bash -c 'mpiexec -n {0} python {1}terngrad_adam.py --max_it {2} --max_t {3} --block_size {4} --lr {5} --beta1 {6} --beta2 {7} --dataset {8} {9}'".format(
                n_workers + 1,
                code_path,
                max_it,
                max_t,
                block_size,
                lr,
                beta1,
                beta2,
                dataset,
                '-b' if big_regularization else ''
            ))


def read_logs(p_norm, alpha, batch, block_size, big_regularization, lr, logs_path='/', method='svrg'):
    alpha *= 1.
    lr *= 1.
    input_params = (
        method,
        p_norm,
        alpha,
        batch,
        block_size,
        'big' if big_regularization else 'small',
        lr
    )

    ws = np.load(logs_path + 'iterates_diana_{0}_{1}_{2}_{3}_{4}_{5}_{6}.npy'.format(*input_params))
    loss = np.load(logs_path + 'loss_diana_{0}_{1}_{2}_{3}_{4}_{5}_{6}.npy'.format(*input_params))
    its = np.load(logs_path + 'iteration_diana_{0}_{1}_{2}_{3}_{4}_{5}_{6}.npy'.format(*input_params))
    ts = np.load(logs_path + 'time_diana_{0}_{1}_{2}_{3}_{4}_{5}_{6}.npy'.format(*input_params))
    return ws, loss, its, ts


def read_logs_qsvrg(batch, block_size, big_regularization, lr, logs_path='/'):
    lr *= 1.
    input_params = (
        batch,
        block_size,
        'big' if big_regularization else 'small',
        lr
    )
    ws = np.load(logs_path + 'iterates_qsvrg_{0}_{1}_{2}_{3}.npy'.format(*input_params))
    loss = np.load(logs_path + 'loss_qsvrg_{0}_{1}_{2}_{3}.npy'.format(*input_params))
    its = np.load(logs_path + 'iteration_qsvrg_{0}_{1}_{2}_{3}.npy'.format(*input_params))
    ts = np.load(logs_path + 'time_qsvrg_{0}_{1}_{2}_{3}.npy'.format(*input_params))
    return ws, loss, its, ts


def read_logs_adam(block_size, beta1, beta2, big_regularization, lr, logs_path='/'):
    input_params = (
        block_size,
        beta1,
        beta2,
        'big' if big_regularization else 'small',
        lr
    )
    ws = np.load(logs_path + 'iterates_adam_{0}_{1}_{2}_{3}_{4}.npy'.format(*input_params))
    loss = np.load(logs_path + 'loss_adam_{0}_{1}_{2}_{3}_{4}.npy'.format(*input_params))
    its = np.load(logs_path + 'iteration_adam_{0}_{1}_{2}_{3}_{4}.npy'.format(*input_params))
    ts = np.load(logs_path + 'time_adam_{0}_{1}_{2}_{3}_{4}.npy'.format(*input_params))
    return ws, loss, its, ts