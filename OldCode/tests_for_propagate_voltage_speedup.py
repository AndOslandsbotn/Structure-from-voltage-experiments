import numpy as np
from utilities.matrices import construct_Wtilde_matrix
from utilities.util import get_nn_indices
from time import perf_counter
from utilities.data_generators import sample_2d_unit_square
from utilities.voltage_solver import apply_voltage_constraints

from scipy.sparse.linalg import svds, eigs
import numpy as np
from time import perf_counter
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.metrics import mean_squared_error


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ###############################################
    config = {
        'kernelType': 'radial_scaled',
        'max_iter': 500,
        'is_Wtilde': True
    }

    # Plane specifications
    d = 2
    D = 4
    eps = 10 ** (-3)

    # Variables
    bw = 0.1  # Bandwidth
    rhoG = 1.e-5  # Inverse of resistance to ground
    rs = 0.2  # Source radius

    tol = 10 ** (-6)
    max_iter = 500
    ###############################################

    # Load plane
    n = 2 ** 12
    plane = sample_2d_unit_square(n, eps=10 ** (-3))
    lm = np.array([0, 0])

    source_indices, _ = get_nn_indices(plane, lm.reshape(1, -1), rs)
    source_indices = list(source_indices[0])

    matrix = construct_Wtilde_matrix(plane, n, source_indices, bw, rhoG, config)

    v = np.zeros(n + 1)
    v = apply_voltage_constraints(v, source_indices)

    # reduce matrix
    matrix = matrix[0:-1, 0:-1]
    v = v[0:-1]
    matrix_sparse = csc_matrix(matrix)

    start = perf_counter()
    i = 0
    vprev = v
    progress = np.inf
    while i < max_iter and progress > tol:
        v = matrix.dot(v)

        i += 1
        progress = mean_squared_error(vprev, v)  # np.sqrt(np.sum((vprev-v)**2))/len(vprev)
        vprev = v
    print("iteration ", i)
    print("progress: ", progress)

    print("TIME standard: ", perf_counter() - start)
    vstandard = v
    # u, e, v = np.linalg.svd(matrix)
    e, u = np.linalg.eig(matrix)
    evec = np.real(u[:, 0])
    evec = -evec

    check_if_evector = np.dot(matrix, evec) / e[0]

    # evec_standard_2 = np.real(u[:, 1])

    start = perf_counter()
    v0 = np.zeros(v.shape[0])
    v0[source_indices] = 1
    us, ss, vhs = svds(matrix_sparse, k=1, v0=v0, which='LM')
    vsparse = us[:, 0]
    vsparse = np.real(vsparse.flatten())

    ss2, us2 = eigs(matrix_sparse, k=1, v0=v0, which='LM')
    evec_sparse = us2[:, 0]
    evec_sparse = np.real(evec_sparse.flatten())
    evec_sparse = - evec_sparse
    print("TIME sparse: ", perf_counter() - start)

    evec_shifted = - evec + np.ones(len(evec))
    shift = 1 - evec_sparse[source_indices[0]]
    evec_sparse_shifted = evec_sparse + shift

    # Visualize
    evec_sparse_sort = np.sort(evec_sparse_shifted, axis=0)
    evec_sort = np.sort(evec_shifted, axis=0)
    v_sort = np.sort(v, axis=0)

    plt.figure()
    plt.plot(evec_sparse_sort, label=f'e-vector sparse')
    plt.plot(evec_sort, label=f'e-vector')
    plt.plot(v_sort, label=f'voltage function')
    plt.legend()

    plt.figure()
    plt.plot(evec_sparse_sort, label=f'e-vector sparse')
    plt.plot(evec_sort, label=f'e-vector')
    plt.legend()

    plt.figure()
    plt.plot(evec_sparse_sort, label=f'e-vector sparse')
    plt.plot(evec_sort, label=f'e-vector')
    plt.plot(v_sort, label=f'voltage function')
    plt.legend()
    plt.yscale('log')
    plt.show()
