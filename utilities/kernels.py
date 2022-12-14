import numpy as np
from scipy.spatial.distance import cdist
from utilities.util import get_nn_indices

def select_kernel(x, n, bw,  kernel_type):
    """Selects a kernel of kerne_type
    :param x: samples on which to construct the kernel
    :param n: Number of sampels
    :param bw: badnwidth of kernel
    """
    if kernel_type == 'radial':
        kernel = radial_kernel(x, bw, n)
    elif kernel_type == 'radial_scaled':
        kernel = radial_kernel_scaled(x, bw, n)
    elif kernel_type == 'gaussian':
        kernel = gaussian_kernel(x, x, bw)
    elif kernel_type == 'gaussian_scaled':
        kernel = gaussian_kernel_scaled(x, x, bw, n)
    return kernel

# Gaussian kernel
def broadcastL2Norm(X1, X2):
    X1 = np.expand_dims(X1, axis=2)
    X2 = np.expand_dims(X2.T, axis=0)
    D = np.linalg.norm(X1 - X2, ord=2, axis=1) ** 2
    return D

def gaussian_kernel(x1, x2, bandwidth, factor=None):
    bandwidth = bandwidth
    if factor != None:
        bandwidth = factor * bandwidth

    D = cdist(x1, x2, metric='sqeuclidean')

    D = (-1 / (2 * bandwidth ** 2)) * D
    return np.exp(D)

def gaussian_kernel_scaled(x1, x2, bandwidth, n, factor=None):
    return (1/n**2)*gaussian_kernel(x1, x2, bandwidth, factor)

# Radial kernel
def radial_kernel(x, r, n):
    nn_indices, _ = get_nn_indices(x, x, r)
    W = np.zeros((n, n))
    for i in range(0, n):
        W[i, list(nn_indices[i])] = 1
    return W

def radial_kernel_scaled(x, r, n):
    return (1/(n**2))*radial_kernel(x, r, n)