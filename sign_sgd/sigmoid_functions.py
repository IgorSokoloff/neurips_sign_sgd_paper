import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import warnings

def sigmoid (x):
    return np.true_divide (1, (1 + np.exp(-x)))

def sigmoid_grad (x):
    #warnings.filterwarnings('error')
    print ("np.exp(-x): {0}".format(np.exp(-x)))
    print("np.power((1 + np.exp(-x)), 2): {0}".format(np.power((1 + np.exp(-x)), 2)))

    try:
        grad = np.true_divide(np.exp(-x), np.power((1 + np.exp(-x)), 2))
    except Warning:
        print ('Warning was raised as an exception!')

    return grad

def reg_bin_clf_loss(w, X, y):
    assert y.shape[0] == X.shape[0]
    assert w.shape[0] == X.shape[1]
    return np.mean(np.power((1 - y * np.exp(sigmoid(X.dot(w)))),2))

def reg_bin_clf_grad(w, X, y):
    assert (len(y) == X.shape[0])
    assert (len(w) == X.shape[1])
    loss_grad = np.mean([reg_bin_clf_sgrad(w, X[i], y[i]) for i in range(len(y))], axis=0)
    assert len(loss_grad) == len(w)
    return loss_grad

def reg_bin_clf_sgrad(w, x_i, y_i):

    assert len(w) == len(x_i)
    assert y_i in [-1, 1]
    loss_sgrad = np.true_divide(2 * y_i * x_i * np.exp(x_i @ w) * (np.exp(x_i @ w)*(y_i - 1) - 1), np.power(1 + np.exp(x_i @ w), 3))
    assert len(loss_sgrad) == len(w)
    return loss_sgrad

def sample_reg_bin_clf_sgrad(w, X, y, batch=1):
    n, d = X.shape
    assert(len(w) == d)
    assert(len(y) == n)
    grad_sum = np.zeros(d)
    for b in range(batch):
        i = random.randint(0, n - 1)
        grad_sum += reg_bin_clf_sgrad(w, X[i], y[i])
    return grad_sum / batch