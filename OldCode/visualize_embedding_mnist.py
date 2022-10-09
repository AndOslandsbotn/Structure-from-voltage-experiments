from utilities.color_maps import color_map, color_map_for_mnist, color_map_list

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import Rectangle, gca
import matplotlib.pyplot as PLT
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import os
import csv
import numpy as np

def visualize_landmarks(landmarks):
    for lm in landmarks:
        lm = lm.reshape(28, 20)
        plt.figure()
        plt.imshow(lm, cmap='gray')

def plot_images_xy_plane(mds_emb, mnist, target, idx_lms, idx_sources, nlm):
    cmap = color_map()
    cmap_mnist = color_map_for_mnist()

    fig = PLT.gcf()
    fig.clf()
    ax = PLT.subplot(111)

    plt.scatter(mds_emb[:, 0], mds_emb[:, 1], s=4, facecolor=cmap['color_darkgray'], lw = 0)
    for idx in idx_sources:
        plt.scatter(mds_emb[idx, 0], mds_emb[idx, 1], s=5, lw = 0, facecolor=cmap['color_darkgray'])

    # select some images to show
    indices = np.random.choice(np.arange(0, len(mnist)), size=np.min([1000, len(mnistdata)]), replace=False)
    for idx in indices:
        digit = int(target[idx])
        mnist_lm = mnist[idx]
        mnist_lm = mnist_lm.reshape(28, 28)
        imagebox = OffsetImage(mnist_lm, zoom=0.5, cmap=cmap_mnist[digit])
        xy = [mds_emb[idx, 0], mds_emb[idx, 1]]  # coordinates to position this image

        ab = AnnotationBbox(imagebox, xy,
                        xybox=(1., -1.),
                        xycoords='data',
                        frameon = False,
                        boxcoords="offset points")
        ax.add_artist(ab)

    # Add source landmarks
    for idx in idx_lms:
        digit = int(target[idx])
        mnist_lm = mnist[idx]
        mnist_lm = mnist_lm.reshape(28, 28)
        imagebox = OffsetImage(mnist_lm, zoom=1.5, cmap=cmap_mnist[digit])
        xy = [mds_emb[idx, 0], mds_emb[idx, 1]]  # coordinates to position this image

        ab = AnnotationBbox(imagebox, xy,
                        xybox=(1., -1.),
                        xycoords='data',
                        frameon=False,
                        boxcoords="offset points")
        ax.add_artist(ab)

    plt.xticks(color='w')
    plt.yticks(color='w')
    ax.set_facecolor('black')
    ax.add_artist(ab)

    # rest is just standard matplotlib boilerplate
    ax.grid(True)
    PLT.draw()
    PLT.show()
    fig.savefig(os.path.join('Figures', f'mnist_embedding_nlm{nlm}.png'))
    fig.savefig(os.path.join('Figures', f'mnist_embedding_nlm{nlm}.eps'), format='eps')

def load_mnist_embedding(filepath):
    #embedding_freyfaces = np.load(os.path.join(filepath, 'embedding_freyfaces.npz'))

    with np.load(os.path.join(filepath, 'embedding_mnist.npz')) as data:
        ff = data['mnistdata']
        target = data['target']
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
    return ff, target, mds_emb, v_emb, idx_lm, idx_sources


if __name__ == '__main__':
    exp_nr = 2
    digit_types = [4]  # Choose a digit type to study

    num_lm_per_digit = 5
    num_digits = len(digit_types)
    nlm = num_digits * num_lm_per_digit

    filepath = os.path.join('Results', f'MnistNlmPerDigit{num_lm_per_digit}_expnr{exp_nr}')
    mnistdata, target, mds_emb, v_emb, idx_lms, idx_sources = load_mnist_embedding(filepath)

    source_points = []
    for indices in idx_sources:
        source_points.append(mnistdata[indices])

    landmarks = []
    for idx in idx_lms:
        landmarks.append(mnistdata[idx])

    v_sort = np.sort(v_emb, axis= 0)
    cmap_list = color_map_list()
    plt.figure()
    for i in range(nlm):
        j = i % 10
        if i < 10:
            plt.plot(v_sort[:, i], label=f'Digit {j}', color = cmap_list[j])
        else:
            plt.plot(v_sort[:, i], color = cmap_list[j])
    plt.legend()
    plt.savefig(os.path.join('Figures', f'mnist_vdecay_NlmPerDigit{num_lm_per_digit}.png'))

    plt.figure()
    for i in range(nlm):
        j = i % 10
        if i < 10:
            plt.plot(v_sort[:, i], label=f'Digit {j}', color = cmap_list[j])
        else:
            plt.plot(v_sort[:, i], color = cmap_list[j])
    plt.yscale('log')
    plt.legend()

    #visualize_landmarks(landmarks)

    plt.figure()
    plt.scatter(mds_emb[:, 0], mds_emb[:, 1])
    for idx in idx_sources:
        plt.scatter(mds_emb[idx, 0], mds_emb[idx, 1])

    plot_images_xy_plane(mds_emb, mnistdata, target, idx_lms, idx_sources, nlm)
    plt.show()