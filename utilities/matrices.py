from utilities.kernels import select_kernel
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix

def construct_w_matrix(x, rhoG, config):
    """Creates a sparse representation of the normalized adjecency matrix (weight matrix) with the ground node
    Parameters
    :param x: n x d vector of n examples of dimension d
    :param rhoG: inverse of resistance to ground
    :param config: configurations
    :return: Weight matrix """

    n = x.shape[0]
    W = np.zeros((n + 1, n + 1))
    if config['metricGraph']['kernelType'] == 'gaussian_scaled' or config['metricGraph']['kernelType'] == 'radial_scaled':
        W[-1, :] = rhoG / n  # To ground
        W[:, -1] = rhoG / n  # To ground
    else:
        W[-1, :] = rhoG  # To ground
        W[:, -1] = rhoG  # To ground

    W[0:-1, 0:-1] = select_kernel(x, n, config['metricGraph']['bandwidth'], config['metricGraph']['kernelType'])
    D = np.sum(W, axis=1)
    D = np.diag(D)
    Dinv = inv(csc_matrix(D))
    Wmatrix = Dinv.dot(W)
    return csc_matrix(Wmatrix)

def construct_wtilde_matrix(x, source_indices, rhoG, config):
    """Constructs the W-tilde matrix, by including the voltage constraints on the source and ground nodes
    Parameters
    :param x: n x d vector of n examples of dimension d
    :param source_indices: indices of the source nodes in x
    :param rhoG: inverse of resistance to ground
    :param config: configurations
    :return: Weight matrix where the source and ground constraints on the voltage are included
    """
    Wtilde = construct_w_matrix(x, rhoG, config)
    Wtilde[source_indices, :] = 0
    Wtilde[source_indices, source_indices] = 1
    Wtilde[-1, :] = 0
    Wtilde[-1, -1] = 1
    return Wtilde