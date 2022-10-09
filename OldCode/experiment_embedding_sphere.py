from utilities.data_generators import sample_data_uniform_sphere
from utilities.embedding import voltage_embedding, multi_dim_scaling, orth_procrustes_edm_to_x
from utilities.sphere_exp_util import get_sphere_section, place_landmarks, generate_longitudes_latitudes

import numpy as np
import os
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

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

def save_experiment(ss, mds_emb, v_emb, idx_lm, idx_lat, idx_long, folder):
    filepath = os.path.join('Results', folder)
    Path(filepath).mkdir(parents=True, exist_ok=True)

    np.savetxt(os.path.join(filepath, 'sphere_segment'), ss, delimiter=",")
    np.savetxt(os.path.join(filepath, 'mds_emb'), mds_emb, delimiter=",")
    np.savetxt(os.path.join(filepath, 'v_emb'), v_emb, delimiter=",")

    with open(os.path.join(filepath, 'idx_lm'), 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows(idx_lm)
    with open(os.path.join(filepath, 'idx_lat'), 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows(idx_lat)
    with open(os.path.join(filepath, 'idx_long'), 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows(idx_long)

if __name__ == '__main__':
    ###############################################
    theta_max  = 90
    theta_min = 0
    phi_max = 180
    phi_min = 0
    step = 10
    limits = {'theta_min': theta_min,
              'theta_max': theta_max,
              'phi_min': phi_min,
              'phi_max': phi_max
              }

    config = {
        'kernelType': 'radial_scaled',
        'max_iter': 1000,
        'is_Wtilde': False
    }

    # Sphere specifications
    d = 3
    eps = 10 ** (-4)
    n = 2 ** 15

    # Variables
    bw = 0.1  # Bandwidth
    rhoG = 1.e-5  # Inverse of resistance to ground
    rs = 0.2  # Source radius

    # Number of landmarks (sources)
    nlm = 5

    ExpFolder = f'SphereTheta{theta_max}Phi{phi_max}nlm{nlm}'
    ###############################################

    theta_grid = np.arange(theta_min+step, theta_max+step, 10)
    phi_grid = np.arange(phi_min, phi_max+step, 10)

    sphere = sample_data_uniform_sphere(n, d, eps)
    sphere_section = get_sphere_section(sphere, limits)
    idx_lat, idx_long = generate_longitudes_latitudes(sphere_section, theta_grid, phi_grid)

    lms = place_landmarks(nlm, limits)
    n_ss = len(sphere_section)
    v_embedding, source_indices = voltage_embedding(sphere_section, lms, n_ss, bw, rs, rhoG,
                                                    config, is_visualization=True)
    mds_embedding = multi_dim_scaling(v_embedding)

    # Use orthogonal procrustes analysis to rotate and translate the MDS embedding to best fit the original orientation
    mds_embedding_rot = orth_procrustes_edm_to_x(sphere_section.transpose(), mds_embedding[:-1, :-1].transpose(), len(sphere_section), 6)
    mds_embedding_rot = mds_embedding_rot.transpose()

    save_experiment(sphere_section, mds_embedding[:, 0:3], v_embedding, source_indices, idx_lat, idx_long, folder=ExpFolder)
    save_experiment(sphere_section, mds_embedding_rot, v_embedding, source_indices, idx_lat, idx_long, folder=ExpFolder)

    # Visualize
    # Visualize
    v_sort = np.sort(v_embedding, axis=0)
    plt.figure()
    for i in range(nlm):
        plt.plot(v_sort[:, i], label=f'Landmark nr {i}')
    plt.legend()
    plt.savefig(os.path.join('Results', ExpFolder, 'VoltageDecaySphere'))

    plot_domain3D(v_embedding, source_indices, v_embedding[0:n_ss, 0], title='Sphere segment with voltage vector')
    plot_domain3D(mds_embedding, source_indices, v_embedding[0:n_ss, 0], title='Sphere segment with MDS vectors')
    plot_domain3D(mds_embedding_rot, source_indices, v_embedding[0:n_ss-1, 0], title='Sphere segment with rotated MDS vectors')
    plt.show()