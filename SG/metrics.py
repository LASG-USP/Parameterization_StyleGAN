import numpy as np
import tensorflow as tf

from scipy.linalg import sqrtm


def frechet_distance(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fd = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fd


def kernel_distance(act1, act2, num_blocks, max_block_size):
    n = act1.shape[1]

    m = min(min(act1.shape[0], act2.shape[0]), max_block_size)
    t = 0
    for _subset_idx in range(num_blocks):
        x = act2[np.random.choice(act2.shape[0], m, replace=False)]
        y = act1[np.random.choice(act1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_blocks / m
    return float(kid)
