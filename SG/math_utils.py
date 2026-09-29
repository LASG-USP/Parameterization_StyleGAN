# math utilities: some algebra tools and the normalized data mismatch objective function
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def cov(M, D, large=None, filter=np.inf):
    """
    sample covariance matrix using two Nm X N and Nd x N arrays (better for large Nm)
    :param M:
    :param D:
    :param large: calculation for large ensemble size
    :return: CMD = covariance between M and D
    """

    if filter is not None:
        mask = np.any(D == filter, axis=0)
        D = np.delete(D, mask, axis=1)
        M = np.delete(M, mask, axis=1)

    N = M.shape[1]
    if large is None:
        # compute ensembles perturbation matrix
        Mp = M @ (np.eye(N) - np.ones((N, N)) * (1 / N))
        Dp = D @ (np.eye(N) - np.ones((N, N)) * (1 / N))

    else:
        Mp = M - np.tile(M.mean(-1), (N, 1)).T
        Dp = D - np.tile(D.mean(-1), (N, 1)).T

    # compute sample covariance (using N-1)
    CMD = (Mp @ Dp.T) / (N - 1)

    return CMD


def corcov(C):
    """
    compute correlation matrix from covariance matrix
    :param C: covariance matrix
    :return: Cor = correlation matrix
    """
    v = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    Cor = C / outer_v
    Cor[C == 0] = 0
    return Cor


def corr(M, D):
    varM = np.var(M, axis=-1, ddof=1)
    varD = np.var(D, axis=-1, ddof=1)
    lower = np.multiply(np.tile(varM, (D.shape[0], 1)).T,
                        np.tile(varD, (M.shape[0], 1)))
    Cmd = cov(M, D)

    out = np.divide(Cmd, np.sqrt(lower), out=np.zeros_like(Cmd), where=lower != 0)
    return out


def normalized_data_mismatch(D, Dobs, Cd):
    icd = np.diag(np.diag(Cd) ** -1)
    N = D.shape[1]
    ond = np.zeros(N)
    # Dobs = np.expand_dims(Dobs, axis=1)
    for n in range(N):
        ond[n] = 1 / (2 * Dobs.shape[0]) * (D[:, [n]] - Dobs).transpose() @ icd @ (D[:, [n]] - Dobs)
    return ond


def sum_normalized_variance(Ma, M):

    M_var = np.var(M, axis=-1, ddof=1)
    Ma_var = np.var(Ma, axis=-1, ddof=1)

    SNV = np.sum(Ma_var/M_var) / M.shape[0]

    return SNV


def root_mean_squared_error(M, ref, axs):
    Nm, N = M.shape

    # rmse = np.zeros(N)
    # for n in range(N):
    #     rmse[n] = np.linalg.norm(M[:, n] - ref) / np.sqrt(Nm)
    #     rmse[n] = np.linalg.norm(M[:, n] - ref) / np.sqrt(Nm)

    rmse = np.linalg.norm(M-ref, axis=axs)/np.sqrt(M.shape[axs])

    return rmse


def vectorization(x, mode='vec', dim=(41, 41)):
    if mode == 'vec':
        if x.ndim > 2:
            N = x.shape[0]
        else:
            N = 1
        y = np.reshape(x.T, (np.prod(dim), N), order='F')
    if mode == 'unvec':
        N = x.shape[-1]
        y = x.reshape((dim[0], dim[1], N), order='F').T
    return y


def balanced_acc(M, ref, axs=0):
    # Calculando a matriz de confusão
    # acc_balanced_per_output = [balanced_accuracy_score(np.round(ref[:, 0]), np.round(M)[:, i]) for i in range(M.shape[1])]
    # return acc_balanced_per_output
    ref = np.round(ref)
    M = np.round(M)
    comparison = M == ref
    # return comparison
    acc = np.sum(comparison, axis=axs)/M.shape[axs]
    return acc