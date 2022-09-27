from utilities.kernels import select_kernel
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix

def construct_W_matrix(x, n, bw, rhoG, config):
    """Creates a sparse representation of the normalized adjecency matrix (weight matrix) with the ground node
    Parameters
    :param x: n x d vector of n examples of dimension d
    :param n: number of examples
    :param bw: bandwidth of kernel matrix
    :param rhoG: inverse of resistance to ground
    :param config: configurations
    :return: Weight matrix """
    Wtemp = select_kernel(x, n, bw, config)
    W = np.zeros((n + 1, n + 1))
    if config['kernelType'] == 'gaussian_scaled' or config['kernelType'] == 'radial_scaled':
        W[-1, :] = rhoG / n  # To ground
        W[:, -1] = rhoG / n  # To ground
    else:
        W[-1, :] = rhoG  # To ground
        W[:, -1] = rhoG  # To ground

    W[0:-1, 0:-1] = Wtemp
    D = np.sum(W, axis=1)
    D = np.diag(D)
    Dinv = inv(csc_matrix(D))
    Wmatrix = Dinv.dot(W)
    return Wmatrix

def construct_Wtilde_matrix(x, n, source_indices, bw, rhoG, config):
    """Constructs the W-tilde matrix, by including the voltage constraints on the source and ground nodes
    Parameters
    :param x: n x d vector of n examples of dimension d
    :param n: number of examples
    :param source_indices: indices of the source nodes in x
    :param bw: bandwidth of kernel matrix
    :param rhoG: inverse of resistance to ground
    :param config: configurations
    :return: Weight matrix where the source and ground constraints on the voltage are included
    """
    Tmatrix = construct_W_matrix(x, n, bw, rhoG, config)
    Tmatrix[source_indices, :] = 0
    Tmatrix[source_indices, source_indices] = 1
    Tmatrix[-1, :] = 0
    Tmatrix[-1, -1] = 1
    return Tmatrix
