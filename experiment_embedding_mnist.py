import numpy as np
import os
import csv

from pathlib import Path
from utilities.mnist_exp_util import load_mnist, pre_processing, select_landmarks_mnist_standard, organize_digits
from utilities.embedding import voltage_embedding, multi_dim_scaling
from utilities.data_io import save_config

import matplotlib
import matplotlib.pyplot as plt


def save_experiment(data, target, mds_emb, v_emb, idx_lms, idx_sources, folder):
    print("Save experiment")
    filepath = os.path.join('Results', folder)
    Path(filepath).mkdir(parents=True, exist_ok=True)

    np.savez(os.path.join(filepath, 'embedding_mnist.npz'), mnistdata=data, target=target, mds_emb=mds_emb, v_emb=v_emb)

    with open(os.path.join(filepath, 'idx_lm'), 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows(idx_lms)

    with open(os.path.join(filepath, 'idx_sources'), 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows(idx_sources)


if __name__ == '__main__':
    #####################
    exp_nr = 1

    config = {
        'kernelType': 'radial_scaled',
        'max_iter': 1000,
        'is_Wtilde': False
    }

    # Variables
    bw = 8  # Bandwidth
    rhoG = 1.e-3  # Inverse of resistance to ground
    rs = 8  # Source radius

    num_lm_per_digit = 1
    nlm = 10 * num_lm_per_digit
    datasize = 500

    folder = f'MnistNlmPerDigit{num_lm_per_digit}_expnr{exp_nr}'
    filepath = os.path.join('Results', folder)
    Path(filepath).mkdir(parents=True, exist_ok=True)
    ######################

    mnistdata, target = load_mnist()
    mnistdata, target = pre_processing(datasize, mnistdata, target)

    digit_indices = organize_digits(target)
    n_mnist = len(mnistdata)
    lms, idx_lms = select_landmarks_mnist_standard(mnistdata, digit_indices, num_lm_per_digit)

    v_embedding, source_indices = voltage_embedding(mnistdata, lms, n_mnist, bw, rs, rhoG, config)
    mds_embedding = multi_dim_scaling(v_embedding)

    save_experiment(mnistdata, target, mds_embedding, v_embedding, idx_lms, source_indices, folder)
    save_config(folder, config, bw, rhoG, rs)

    v_sort = np.sort(v_embedding, axis=0)
    plt.figure()
    for i in range(nlm):
        plt.plot(v_sort[:, i], label=f'Digit nr {i}')
    plt.legend()
    plt.savefig(os.path.join('Results', folder, 'VoltageDecayMnist'))
