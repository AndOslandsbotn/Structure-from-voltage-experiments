from tqdm import tqdm

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
    :param is_Wtilde: Whether the matrix is W or Wtilde
    """
    for _ in range(0, max_iter):
        if is_Wtilde == True:
            v = matrix.dot(v)
        else:
            v = matrix.dot(v)
            v = apply_voltage_constraints(v, source_indices)
    if is_visualization:
        # For visualization purposes we apply the propagation 1 more time without setting source to 1
        v = matrix.dot(v)
    return v
