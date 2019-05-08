import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

supported_penalties = ['l1', 'l2']

def logreg_loss(x, A, y, la):
    assert la >= 0
    assert len(y) == A.shape[0]
    assert len(x) == A.shape[1]
    l = np.log(1 + np.exp(-A.dot(x) * y))
    m = y.shape[0]
    return np.sum(l) / m + la/2 * norm(x) ** 2

def logreg_grad(w, X, y, la):
    assert la >= 0
    assert (len(y) == X.shape[0])
    assert (len(w) == X.shape[1])
    loss_grad = np.mean([logreg_sgrad(w, X[i], y[i]) for i in range(len(y))], axis=0)
    assert len(loss_grad) == len(w)
    return loss_grad + la * w

def logreg_sgrad(w, x_i, y_i, la=0):
    assert la >= 0
    assert len(w) == len(x_i)
    assert y_i in [-1, 1]
    loss_sgrad = - y_i * x_i / (1 + np.exp(y_i * np.dot(x_i, w)))
    assert len(loss_sgrad) == len(w)
    return loss_sgrad + la * w

def sample_logreg_sgrad(w, X, y, mu=0, batch=1):
    assert mu >= 0
    n, d = X.shape
    assert(len(w) == d)
    assert(len(y) == n)
    grad_sum = 0
    for b in range(batch):
        i = random.randrange(n)
        grad_sum += logreg_sgrad(w, X[i], y[i], mu)
    return grad_sum / batch

def f(x, A, y, la):
    assert ((y.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert (la >= 0)
    return logreg_loss(x, A, y, la)

