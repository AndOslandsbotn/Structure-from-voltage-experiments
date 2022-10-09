import numpy as np
from scipy.spatial.distance import cdist

def multi_dim_scaling(x):
    """ Make multi-dimensional scaling embedding of x
    Parameters
    ----------
    :param x: Coordinates as n x d numpy array, where n is number of training examples and d is the dimension
    :return: euclidean distance matrix n x n numpy array
    """
    voltages_centered = x - np.mean(x, axis =0)
    u, sigma, vh = np.linalg.svd(voltages_centered)
    s_temp = np.zeros(len(x))
    s_temp[0:len(sigma)] = sigma[0:len(sigma)]
    sigma = s_temp
    x_mds= np.dot(u, np.diag(sigma))
    return x_mds
