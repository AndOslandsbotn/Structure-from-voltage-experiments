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

def load_plane(folder):
    filepath = os.path.join('Results', folder)

    with np.load(os.path.join(filepath, 'embedding_plane.npz')) as data:
        plane = data['plane']
        mds_emb = data['mds_emb']
        v_emb = data['v_emb']

    with open(os.path.join(filepath, 'idx_sources')) as csv_file:
        csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        idx_sources = []
        for row in csv_reader:
            row = [int(e) for e in row] # Convert to int
            idx_sources.append(row)
    return plane, mds_emb, v_emb, idx_sources

if __name__ == '__main__':
    # Plane specifications
    d = 2
    D = 4
    nlm = 4
    folder = f'PlaneD{D}d{d}Nlm{nlm}'
    plane, mds_embedding, v_embedding, source_indices = load_plane(folder)

    # Visualize
    v_sort = np.sort(v_embedding, axis=0)
    plt.figure()
    for i in range(nlm):
        plt.plot(v_sort[:, i], label=f'Landmark nr {i}')
    plt.legend()
    plt.savefig(os.path.join('Results', folder, 'VoltageDecayPlane'))

    radius = np.sqrt(np.sum(plane, axis=1))
    plot_domain3D(v_embedding, source_indices, radius, title='Unit square voltage embedding')
    plot_domain3D(mds_embedding, source_indices, radius, title='Unit square mds embedding')
    plt.show()
