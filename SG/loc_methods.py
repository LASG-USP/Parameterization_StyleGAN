import numpy as np
from math_utils import cov


def compute_furrer_localization(x, n, type='full', y=None, ddof=0):
    if type == 'full':
        C = np.cov(x, ddof=ddof)
        varx = np.diag(C)
        vary = np.diag(C)
    elif type == 'cross':
        C = cov(x, y)
        varx = np.var(x, axis=-1, ddof=ddof)
        vary = np.var(y, axis=-1, ddof=ddof)
    P = np.zeros(C.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] != 0:
                up = C[i, j] ** 2
                bo = C[i, j] ** 2 + (C[i, j] ** 2 + varx[i]*vary[j])/ n
                P[i, j] = max(up / bo, 0)
    return P

def compute_scaling_approximate(x, n, type='full', y=None, ddof=0, norm=None):
    if type == 'full':
        C = np.cov(x, ddof=ddof)
        tr2c = np.trace(C)**2
    elif type == 'cross':
        if norm is not None:
            y = np.diag(1 / np.sqrt(norm.diagonal())) @ y
        C = cov(x, y)
        tr2c = np.sum(np.var(x, axis=-1, ddof=ddof))*np.sum(np.var(y, axis=-1, ddof=ddof))

    trc2 = np.trace(C @ C.T)
    return max(0, (trc2 - tr2c/(n-ddof)) / (trc2+trc2/(n-ddof)))