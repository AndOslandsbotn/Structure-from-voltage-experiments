import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utilities.embedding import voltage_embedding, multi_dim_scaling
from utilities.data_generators import unit_square_domain

def plot_domain3D(x, lm_indices_all_lm, radius, title):
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:-1, 0], x[:-1, 1], x[:-1, 2], c=radius)
    for lm_indices in lm_indices_all_lm:
        ax.scatter(x[lm_indices, 0],
                    x[lm_indices, 1],
                    x[lm_indices, 2],
                    marker='s', c='red')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.view_init(-0, 0)

if __name__ == '__main__':
    ###############################################
    config = {
        'kernelType': 'radial_scaled',
        'max_iter': 300,
        'is_Wtilde': False
    }

    # Plane specifications
    d = 2
    D = 4
    eps = 10 ** (-3)
    N = 1000

    # Variables
    bw = 0.1  # Bandwidth
    rhoG = 1.e-5  # Inverse of resistance to ground
    rs = 0.2  # Source radius

    # Embedding dim for the multi-dim scaling analysis
    mds_embedding_dim = 3
    ###############################################

    # Load plane
    n = 2**12
    plane = unit_square_domain(n, eps=10**(-3))

    # Make embedding
    lms = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    voltage_embedding, source_indices = voltage_embedding(plane, lms, n, bw, rs, rhoG, config)

    mds_embedding = multi_dim_scaling(voltage_embedding, mds_embedding_dim)

    # Visualize
    radius = np.sqrt(np.sum(plane, axis=1))
    plot_domain3D(voltage_embedding, source_indices, radius, title='Unit square voltage embedding')
    plot_domain3D(mds_embedding, source_indices, radius, title='Unit square mds embedding')
    plt.show()