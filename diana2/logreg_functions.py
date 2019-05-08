import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

supported_penalties = ['l1']

def logreg_loss(x, A, b, l2):
    assert l2 >= 0
    assert len(b) == A.shape[0]
    assert len(x) == A.shape[1]
    l = np.log(1 + np.exp(-A.dot(x) * b))
    m = b.shape[0]
    return np.sum(l) / m + l2/2 * norm(x) ** 2

def logreg_grad(w, X, y, mu):
    assert mu >= 0
    assert (len(y) == X.shape[0])
    assert (len(w) == X.shape[1])
    loss_grad = np.mean([logreg_sgrad(w, X[i], y[i]) for i in range(len(y))], axis=0)
    assert len(loss_grad) == len(w)
    return loss_grad + mu * w

def logreg_sgrad(w, x_i, y_i, mu=0):
    assert mu >= 0
    assert len(w) == len(x_i)
    assert y_i in [-1, 1]
    loss_sgrad = - y_i * x_i / (1 + np.exp(y_i * np.dot(x_i, w)))
    assert len(loss_sgrad) == len(w)
    return loss_sgrad + mu * w

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

def r(x, l1):
    return l1 * norm(x, ord = 1)

def F(x, A, b, l2, l1=0):
    assert ((b.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert ((l2 >= 0) & (l1 >= 0))
    return logreg_loss(x, A, b, l2) + r(x, l1)

def prox_r(x, gamma, coef, penalty='l1'):
    assert penalty in supported_penalties
    assert(gamma > 0 and coef >= 0)
    if penalty == 'l1':
        l1 = coef
        return x - abs(x).minimum(l1 * gamma).multiply(x.sign())