import numpy as np
import os
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from utilities.embedding import voltage_embedding, multi_dim_scaling, euclidean_distance_matrix
from utilities.data_generators import sample_2d_unit_square

def save_experiment(plane, mds_emb, v_emb, idx_sources, folder):
    filepath = os.path.join('Results', folder)
    Path(filepath).mkdir(parents=True, exist_ok=True)

    np.savez(os.path.join(filepath, 'embedding_plane.npz'), plane=plane, mds_emb=mds_emb, v_emb=v_emb)

    with open(os.path.join(filepath, 'idx_sources'), 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows(idx_sources)

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

    # Variables
    bw = 0.1  # Bandwidth
    rhoG = 1.e-5  # Inverse of resistance to ground
    rs = 0.2  # Source radius
    ###############################################

    # Load plane
    n = 2**12
    plane = sample_2d_unit_square(n, eps=10**(-3))

    # Make embedding
    lms = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    v_embedding, source_indices = voltage_embedding(plane, lms, n, bw, rs, rhoG,
                                                    config, is_visualization=True)
    mds_embedding = multi_dim_scaling(v_embedding)

    folder = f'PlaneD{D}d{d}Nlm{len(lms)}'
    save_experiment(plane, mds_embedding, v_embedding, source_indices, folder)
