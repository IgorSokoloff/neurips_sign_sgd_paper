import numpy as np

def linreg_loss(ws, Xs, ys, l2, N=1):
    n = len(ys)
    return [np.sum([0.5 * w @ Xs[i] @ w + ys[i] @ w for i in range (n)]) / N + l2 / 2 * np.linalg.norm(w, ord=2) ** 2 for w in ws]

def logreg_loss(ws, X, y, l1, l2):
    n = len(y)
    if 2 in y:
        y = 2 * y - 3
    return np.array([np.mean(np.log(1 + np.exp(-X @ w * y))) + l2 / 2 * np.linalg.norm(w, ord=2) ** 2 + l1 * np.linalg.norm(w, ord=1) for w in ws])