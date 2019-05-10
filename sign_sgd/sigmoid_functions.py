import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize


def sigmoid_loss(w, A, y, la):
    assert la >= 0
    assert len(y) == A.shape[0]
    assert len(w) == A.shape[1]
    l = np.log(1 + np.exp(-A.dot(w) * y))
    m = y.shape[0]
    return np.sum(l) / m + la/2 * norm(w) ** 2

def sigmoid_grad(w, X, y, la):
    assert la >= 0
    assert (len(y) == X.shape[0])
    assert (len(w) == X.shape[1])
    loss_grad = np.mean([sigmoid_sgrad(w, X[i], y[i]) for i in range(len(y))], axis=0)
    assert len(loss_grad) == len(w)
    return loss_grad + la * w

def sigmoid_sgrad(w, x_i, y_i, la=0):
    assert la >= 0
    assert len(w) == len(x_i)
    assert y_i in [-1, 1]
    loss_sgrad = - y_i * x_i / (1 + np.exp(y_i * np.dot(x_i, w)))
    assert len(loss_sgrad) == len(w)
    return loss_sgrad + la * w

def sample_sigmoid_sgrad(w, X, y, la=0, batch=1):
    assert la >= 0
    n, d = X.shape
    assert(len(w) == d)
    assert(len(y) == n)
    grad_sum = 0
    for b in range(batch):
        i = random.randrange(n)
        grad_sum += sigmoid_sgrad(w, X[i], y[i], la)
    return grad_sum / batch