import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize


def sigmoid (x):
    return np.divide (1, (1 + np.exp(-x)))

def sigmoid_grad (x):
    return np.divide(np.exp(-x), np.power((1 + np.exp(-x)),2))

def reg_bin_clf_loss(w, A, y):
    assert len(y) == A.shape[0]
    assert len(w) == A.shape[1]
    l = np.power((1 - y * np.exp(sigmoid(A.dot(w)))),2)
    m = y.shape[0]
    return np.sum(l) / m

def reg_bin_clf_grad(w, X, y):
    assert (len(y) == X.shape[0])
    assert (len(w) == X.shape[1])
    loss_grad = np.mean([reg_bin_clf_sgrad(w, X[i], y[i]) for i in range(len(y))], axis=0)
    assert len(loss_grad) == len(w)
    return loss_grad

def reg_bin_clf_sgrad(w, x_i, y_i):

    assert len(w) == len(x_i)
    assert y_i in [-1, 1]
    loss_sgrad = 2 *(1 - y_i * sigmoid(np.dot(x_i, w))) *y_i* sigmoid_grad (np.dot(x_i, w)) * x_i
    assert len(loss_sgrad) == len(w)
    return loss_sgrad

def sample_reg_bin_clf_sgrad(w, X, y, batch=1):

    n, d = X.shape
    assert(len(w) == d)
    assert(len(y) == n)
    grad_sum = 0
    for b in range(batch):
        i = random.randrange(n)
        grad_sum += reg_bin_clf_sgrad(w, X[i], y[i])
    return grad_sum / batch