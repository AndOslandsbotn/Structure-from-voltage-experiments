import numpy as np
from scipy.spatial.distance import cdist

from utilities.voltage_solver import apply_voltage_constraints, propagate_voltage
from utilities.matrices import construct_W_matrix
from utilities.util import get_nn_indices
from tqdm import tqdm

def voltage_embedding(x, lms, n, bw, rs, rhoG, config, is_visualization=False):
    """
    Parameters
    ----------
    :param x: Data points n x d numpy array, where n is the number of points, d the dimension
    :param lms: Source landmarks m x d numpy array, where m is the number of landmarks, d the dimension
    :param rs: Source radius
    :return voltages: embedding n x m numpy array
    :return source_indices_l: list of list. Inner list nr i contains
     source indices for all points in distance rs from source landmark nr i.
    """
    voltages = []
    source_indices_l = []
    matrix = construct_W_matrix(x, n, bw, rhoG, config) # Construct the adjacency matrix W
    for lm in tqdm(lms, desc='Loop landmarks'):
        # Get indices of all points in x that are distance $r_s$ from the landmark lm
        source_indices, _ = get_nn_indices(x, lm.reshape(1, -1), rs)
        source_indices = list(source_indices[0])
        source_indices_l.append(source_indices)

        # Initialize a voltage vector, with source and ground constraints applied
        init_voltage = np.zeros(n + 1)
        init_voltage = apply_voltage_constraints(init_voltage, source_indices)

        # Propagate the voltage to all points in the dataset
        voltages.append(propagate_voltage(init_voltage, matrix, config['max_iter'],
                                          source_indices, config['is_Wtilde'],
                                          is_visualization))
    return np.array(voltages).transpose(), source_indices_l

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

# Multi-dim scaling
def distance_matrix(X):
    return np.square(cdist(X.transpose(), X.transpose(), metric='euclidean'))

def gram_from_dist(D, n, s):
    """Calculates gram matrix from distance matrix"""
    I = np.diag(np.ones(n))
    I1 = np.ones(n)
    rhs = (I-np.outer(I1, s))
    lhs = (I-np.outer(s, I1))
    return -1/2*np.dot(rhs, np.dot(D, lhs))

def spectral_cutoff(s, d):
    s_temp = np.zeros(len(s))
    s_temp[0:d] = s[0:d]
    return s_temp

def subsample(X, X_EDM, m):
    index = np.random.choice(X.shape[1], m, replace=False)
    Xw = X[:, index]
    lm_EDM = X_EDM[:, index]
    return Xw, lm_EDM

def center(X):
    c = np.mean(X, axis=1)
    return X - c.reshape(-1, 1), c

def euclidean_distance_matrix(x, s):
    d, n = np.shape(x)
    D = distance_matrix(x)
    G = gram_from_dist(D, n, s)
    u, sigma, uh = np.linalg.svd(G)
    sigma_cut = spectral_cutoff(sigma, d)

    I1 = np.ones(n)
    J = np.diag(np.ones(n)) - np.outer(s, I1)
    x_shifted = np.dot(x, J)
    return np.dot(np.diag(np.sqrt(sigma_cut)), u.transpose()), x_shifted

def orth_procrustes_x_to_edm(x, x_edm, n, m):
    """Implements orthogonal procrustes problem. Rotates x to align with x_edm"""

    x_sub, lm_edm = subsample(x, x_edm, m)

    lm_edm, center_lm_edm = center(lm_edm)
    x_sub, center_x = center(x_sub)

    x_sub_lm_edm = np.dot(x_sub, lm_edm.transpose())
    u, s, vh = np.linalg.svd(x_sub_lm_edm, full_matrices=False)

    R = np.dot(vh.transpose(), u.transpose())
    I1 = np.ones(n)
    x_shift = x - np.outer(center_x, I1)

    return np.dot(R, x_shift) + np.outer(center_lm_edm, I1)


def orth_procrustes_edm_to_x(x, x_edm, n, m):
    """Implements orthogonal procrustes problem. Rotates x_edm to align with x"""

    lm_x, edm_sub = subsample(x, x_edm, m)
    lm_x, center_lm_x = center(lm_x)
    edm_sub, center_edm = center(edm_sub)

    lm_x_edm_sub = np.dot(edm_sub, lm_x.transpose())
    u, s, vh = np.linalg.svd(lm_x_edm_sub, full_matrices=False)

    R = np.dot(vh.transpose(), u.transpose())
    I1 = np.ones(n)
    x_shift = x_edm - np.outer(center_edm, I1)

    return np.dot(R, x_shift) + np.outer(center_lm_x, I1)
