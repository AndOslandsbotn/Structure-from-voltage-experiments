from tqdm import tqdm
from scipy.sparse.linalg import svds, eigs
import numpy as np
from time import perf_counter
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.metrics import mean_squared_error

def apply_voltage_constraints(voltage, source_indices):
    """Apply the source and ground voltage constraints to the voltage vector
    :param v: Voltage vector
    :param source_indices: Indices of source nodes in v
    """
    voltage[source_indices] = 1  # Source voltage
    voltage[-1] = 0  # Ground voltage
    return voltage

def propagate_voltage(v, matrix, max_iter, source_indices, is_Wtilde=False, is_visualization=False):
    """Propagate voltage by iteratively applying the matrix on the voltage vector of
    Parameters
    :param v: Initial voltage vector
    :param matrix: Either W (adjacency matrix) or Wtilde matrix (adjacency matrix with voltage constraints included)
    :param max_iter: maximum iterations
    :param source_indices: Indices of source nodes in v
    :param is_Wtilde: Whether the matrix is standard weight matrix W or weight matrix Wtilde
    where source conditions are enforced
    :param is_visualization: If true we propagation 1 more time without
    setting source to 1, to make visualization prettier
    """

    i = 0
    tol = 10**(-6)
    vprev = v
    progress = np.inf
    while i < max_iter and progress > tol:
        if is_Wtilde == True:
            v = matrix.dot(v)
        else:
            v = matrix.dot(v)
            v = apply_voltage_constraints(v, source_indices)
        i += 1
        progress = mean_squared_error(vprev, v)  # np.sqrt(np.sum((vprev-v)**2))/len(vprev)
        vprev = v

    if is_visualization:
        v = matrix.dot(v)
    return v
