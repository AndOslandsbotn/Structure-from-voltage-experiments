import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pylab import Rectangle, gca

import matplotlib.pyplot as PLT
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import os
import csv

def visualize_landmarks(landmarks):
    for lm in landmarks:
        lm = lm.reshape(28, 20)
        plt.figure()
        plt.imshow(lm, cmap='gray')

def plot_images_xy_plane(mds_emb, ff, idx_lms, idx_sources, nlm):
    color_red = '#CD3333'
    color_yel = '#E3CF57'
    color_gray = '#C1CDCD'
    color_darkgray = '#838B8B'

    fig = PLT.gcf()
    fig.clf()
    ax = PLT.subplot(111)

    plt.scatter(mds_emb[:, 0], mds_emb[:, 1], s=4, facecolor=color_darkgray, lw = 0)
    for idx in idx_sources:
        plt.scatter(mds_emb[idx, 0], mds_emb[idx, 1], s=5, lw = 0, facecolor=color_darkgray)

    # Randomly select some images to show
    indices = np.random.choice(np.arange(0, len(ff)), size=len(ff), replace=False)
    for idx in indices:
        ff_lm = ff[idx]
        ff_lm = ff_lm.reshape(28, 20)
        imagebox = OffsetImage(ff_lm, zoom=0.5, cmap='gray')
        xy = [mds_emb[idx, 0], mds_emb[idx, 1]]  # coordinates to position this image

        ab = AnnotationBbox(imagebox, xy,
                        xybox=(1., -1.),
                        xycoords='data',
                        frameon = False,
                        boxcoords="offset points")
        ax.add_artist(ab)

    # Add source landmarks
    for idx in idx_lms:
        ff_lm = ff[idx]
        ff_lm = ff_lm.reshape(28, 20)
        imagebox = OffsetImage(ff_lm, zoom=0.8, cmap='bone')
        xy = [mds_emb[idx, 0], mds_emb[idx, 1]]  # coordinates to position this image

        ab = AnnotationBbox(imagebox, xy,
                        xybox=(1., -1.),
                        xycoords='data',
                        frameon=False,
                        boxcoords="offset points")
        ax.add_artist(ab)

    ax.add_artist(ab)

    # rest is just standard matplotlib boilerplate
    ax.grid(True)
    PLT.draw()
    PLT.show()
    fig.savefig(os.path.join('Figures', f'freyface_embedding_nlm{nlm}.eps'), format='eps')

def load_freyface_embedding(filepath):
    with np.load(os.path.join(filepath, 'embedding_freyfaces.npz')) as data:
        ff = data['ff_data']
        mds_emb = data['mds_emb']
        v_emb = data['v_emb']

    with open(os.path.join(filepath, 'idx_lm')) as csv_file:
        csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        idx_lm = []
        for row in csv_reader:
            row = [int(e) for e in row] # Convert to int
            idx_lm.append(row)

    with open(os.path.join(filepath, 'idx_sources')) as csv_file:
        csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        idx_sources = []
        for row in csv_reader:
            row = [int(e) for e in row] # Convert to int
            idx_sources.append(row)

    return ff, mds_emb, v_emb, idx_lm, idx_sources

if __name__ == '__main__':
    exp_nr = 2
    nlm = 10

    folder = f'FreyFacesNlm{nlm}_expnr{exp_nr}'
    filepath = os.path.join('Results', folder)
    ff, mds_emb, v_emb, idx_lms, idx_sources = load_freyface_embedding(filepath)

    source_points = []
    for indices in idx_sources:
        source_points.append(ff[indices])

    landmarks = []
    for idx in idx_lms:
        landmarks.append(ff[idx])

    # visualize_landmarks(landmarks)

    plt.figure()
    plt.scatter(mds_emb[:, 0], mds_emb[:, 1])
    for idx in idx_sources:
        plt.scatter(mds_emb[idx, 0], mds_emb[idx, 1])

    plot_images_xy_plane(mds_emb, ff, idx_lms, idx_sources, nlm)
    plt.show()