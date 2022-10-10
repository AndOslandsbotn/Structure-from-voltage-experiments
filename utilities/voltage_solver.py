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

def propagate_voltage(voltage, weight_matrix, source_indices, config):
    """Propagate voltage by iteratively applying the matrix on the voltage vector of
    Parameters
    :param voltage: Initial voltage vector
    :param weight_matrix: adjacency matrix (weight matrix)
    :param source_indices: Indices to all nodes that are radius r_s from a source landmark
    :param config: Configurations
    """
    voltage_prev = voltage
    progress = np.inf
    start = perf_counter()
    for i in tqdm(range(0, config['voltageSolver']['max_iter']), desc='propagating voltage'):
        if progress <= config['voltageSolver']['tol']:
            break
        voltage = weight_matrix.dot(voltage)
        voltage = apply_voltage_constraints(voltage, source_indices)

        progress = mean_squared_error(voltage_prev, voltage)
        voltage_prev = voltage

    if config['general']['is_visualization']:
        voltage = weight_matrix.dot(voltage)
    print("Time to propagate: ", perf_counter()-start)
    return voltage
