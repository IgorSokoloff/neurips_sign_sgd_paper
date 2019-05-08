import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from logreg_functions import *


def svrg(x_init, X, y, eta, mu=0, S=50, M=None, max_t=np.inf):
    n, d = X.shape
    assert (len(x_init) == d)
    assert (len(y) == n)
    if M is None:
        M = 4 * n
    x = np.array(x_init)
    xs = []
    its = []
    ts = []
    t_start = time.time()
    step = min((S * M) // 1000, M // 2)

    for s in range(S):
        full_grad = logreg_grad(x, X, y, mu)
        x0 = np.array(x)
        for it in range(M):
            i = random.randrange(n)
            grad = logreg_sgrad(x, X[i], y[i], mu)
            old_grad = logreg_sgrad(x0, X[i], y[i], mu)
            v = grad - old_grad + full_grad
            x -= eta * v
            if it % step == 0 or (s == 0 and it <= 50):
                xs.append(np.array(x))
                its.append(s * M + it)
                ts.append(time.time() - t_start)
        if ts[-1] > max_t:
            break
    return xs, its, ts