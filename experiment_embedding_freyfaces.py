import numpy as np
import os
import csv
from pathlib import Path

from utilities.freyface_exp_util import select_landmarks_kmeanspp, select_landmarks_max_spread, select_landmarks_standard
from utilities.embedding import voltage_embedding, multi_dim_scaling
from utilities.data_io import save_config

import matplotlib
import matplotlib.pyplot as plt


def save_experiment(data, mds_emb, v_emb, idx_lms, idx_sources, folder):
    print("Save experiment")
    filepath = os.path.join('Results', folder)
    Path(filepath).mkdir(parents=True, exist_ok=True)

    np.savez(os.path.join(filepath, 'embedding_freyfaces.npz'), ff_data=data, mds_emb=mds_emb, v_emb=v_emb)

    with open(os.path.join(filepath, 'idx_lm'), 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows(idx_lms)

    with open(os.path.join(filepath, 'idx_sources'), 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows(idx_sources)

if __name__ == '__main__':
    #####################################################
    exp_nr = 2

    config = {
        'kernelType': 'radial_scaled',
        'max_iter': 1000,
        'is_Wtilde': False
    }

    # Variables
    bw = 2  # Bandwidth
    rhoG = 1.e-3  # Inverse of resistance to ground
    rs = 2.2  # Source radius

    nlm = 10
    Normalize_images = 255

    folder = f'FreyFacesNlm{nlm}_expnr{exp_nr}'
    filepath = os.path.join('Results', folder)
    Path(filepath).mkdir(parents=True, exist_ok=True)
    #######################################################

    # Load frey face data
    with np.load('Data/freydata.npz') as data:
        ff = data['ff_data']
        n, m, t = ff.shape
        ff = ff.reshape(-1, m*t)
        ff = ff/Normalize_images  # Normalize

    n_ff = len(ff)

    # Select landmarks
    #lms = select_landmarks_standard(ff, n, nlm)
    #lms, idx_lms = select_landmarks_max_spread(ff, n, nlm)
    lms, idx_lms = select_landmarks_kmeanspp(ff, n, nlm)

    v_embedding, source_indices = voltage_embedding(ff, lms, n_ff, bw, rs, rhoG,
                                                    config, is_visualization=True)
    mds_embedding = multi_dim_scaling(v_embedding)

    save_experiment(ff, mds_embedding, v_embedding, idx_lms, source_indices, folder)
    save_config(folder, config, bw, rhoG, rs)

    v_sort = np.sort(v_embedding, axis= 0)
    plt.figure()
    for i in range(nlm):
        plt.plot(v_sort[:, i], label=f'v{i}')
    plt.savefig(os.path.join('Results', folder, 'VoltageDecayFF'))


