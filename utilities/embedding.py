import numpy as np

from utilities.voltage_solver import apply_voltage_constraints, propagate_voltage
from utilities.matrices import construct_W_matrix
from utilities.util import get_nn_indices
from tqdm import tqdm

def voltage_embedding(x, lms, n, bw, rs, rhoG, config):
    """
    Parameters
    ----------
    :param x: Data points n x d numpy array, where n is the number of points, d the dimension
    :param lms: Landmarks m x d numpy array, where m is the number of landmarks, d the dimension
    :param rs: Source radius
    :return:
    """
    voltages = []
    source_indices_l = []
    for lm in tqdm(lms, desc='Loop landmarks'):
        source_indices, _ = get_nn_indices(x, lm.reshape(1, -1), rs)
        source_indices = list(source_indices[0])
        source_indices_l.append(source_indices)

        matrix = construct_W_matrix(x, n, bw, rhoG, config)
        init_voltage = np.zeros(n + 1)
        init_voltage = apply_voltage_constraints(init_voltage, source_indices)
        voltages.append(propagate_voltage(init_voltage, matrix, config['max_iter'],
                                          source_indices, config['is_Wtilde']))
    return np.array(voltages).transpose(), source_indices_l

def multi_dim_scaling(x, embedding_dim):
    """ Make multi-dimensional scaling embedding of x
    Parameters
    ----------
    :param x: Coordinates as n x d numpy array, where n is number of training examples and d is the dimension
    :return:
    """
    voltages_centered = x - np.mean(x, axis =0)
    u, sigma, vh = np.linalg.svd(voltages_centered)
    s_temp = np.zeros(len(x))
    s_temp[0:len(sigma)] = sigma[0:len(sigma)]
    sigma = s_temp
    x_mds= np.dot(u, np.diag(sigma))
    return x_mds[:, 0:embedding_dim]