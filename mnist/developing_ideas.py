import numpy as np
import os
import json
import csv
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from utilities.util import flatten_list
from utilities.multidimensional_scaling import multi_dim_scaling
from utilities.color_maps import color_map, color_map_for_mnist, color_map_list

from definitions import CONFIG_PATH, ROOT_DIR
from config.yaml_functions import yaml_loader
config = yaml_loader(CONFIG_PATH)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import Rectangle, gca
import matplotlib.pyplot as PLT
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


class MNIST_embedding_visualization():
    def __init__(self, config, experiment_nr):
        self.config = config
        self.data, \
        self.labels, \
        self.digit_indices, \
        self.landmark_indices_dict, \
        self.source_indices_dict, \
        self.voltages_dict = self.load_mnist_embedding(experiment_nr)
        return

    def get_source_indices(self, level, digits):
        source_indices = []
        for digit in digits:
            source_indices += self.source_indices_dict[f'level{level}'][f'digit{digit}']
        return flatten_list(source_indices)

    def get_landmark_indices(self, level, digits):
        landmark_indices = []
        for digit in digits:
            landmark_indices += self.landmark_indices_dict[f'level{level}'][f'digit{digit}']
        return flatten_list(landmark_indices)

    def calc_mutual_distances(self, x):
        return cdist(x, x)

    def max_distances(self, landmarks):
        distances = self.calc_mutual_distances(landmarks)
        return np.max(distances, axis=1)

    def voltageSpace_convexHull(self, max_dist, voltage_embedding, landmark_indices):
        distances = self.calc_mutual_distances(voltage_embedding)
        dist_to_landmarks = distances[:, landmark_indices]

        bool = dist_to_landmarks > max_dist + self.config['convexHull']['tol']
        inside_hull = np.sum(bool, axis=1) == 0
        indices = np.arange(0, voltage_embedding.shape[0])
        indices_inside_hull = indices[inside_hull]
        return indices_inside_hull

    def visualize_voltageSpace_convexHull(self, level, digits, landmark_numbers):
        landmark_indices = np.array(self.get_landmark_indices(level, digits))[landmark_numbers]
        voltage_embedding = self.get_voltage_embedding(level, digits)[0:-1, :]
        lm_v_emb = voltage_embedding[landmark_indices, :]
        max_dist = self.max_distances(lm_v_emb)
        indices_inside_hull = self.voltageSpace_convexHull(max_dist, voltage_embedding, landmark_indices)

        # Hull using convex hull method in mds space
        #voltage_embedding_in_hull = voltage_embedding[indices_inside_hull, :]
        mds_emb, nlm = self.multi_dim_scaling(voltage_embedding)
        lm_mds_emb = mds_emb[landmark_indices, :]
        points_in_hull_mds_emb = mds_emb[indices_inside_hull, :]
        points_outside_hull_mds_emb = mds_emb[~indices_inside_hull, :]

        hull = self.find_convexHull(lm_mds_emb[:, 0:2])
        self.plotting_points_in_convexHull(lm_mds_emb, points_in_hull_mds_emb, points_outside_hull_mds_emb, hull)

        # Hull using voltage cutof method
        voltage_embedding_in_hull = voltage_embedding[indices_inside_hull, :]
        mds_embedding_in_hull, nlm = self.multi_dim_scaling(voltage_embedding_in_hull)
        mnistdata_in_hull = self.data[indices_inside_hull, :]
        labels_in_hull = self.labels[indices_inside_hull]
        new_landmark_indices = np.where(np.in1d(indices_inside_hull, landmark_indices))[0]

        #self.plot_images_xy_plane(mnistdata_in_hull, labels_in_hull, mds_embedding_in_hull, new_landmark_indices, level, digits, nlm, experiment_name='convexhull_vs')
        #self.plotting_points_in_convexHull(lm_mds_emb, points_in_hull, points_outside_hull, hull)

        # Hull using convex hull method in voltage space where we have 1 extra lm
        voltage_embedding = voltage_embedding[:, landmark_numbers[0:-1]]
        hull = self.find_convexHull(lm_v_emb[:, landmark_numbers[0:-1]])
        is_in_hull = self.find_points_in_convexHull(voltage_embedding, hull)

        lm_mds_emb = mds_emb[landmark_indices, :]

        points_in_hull = mds_emb[is_in_hull, :]
        points_outside_hull = mds_emb[~is_in_hull, :]
        #self.plotting_points_in_convexHull(lm_mds_emb, points_in_hull, points_outside_hull, hull)

    def find_convexHull(self, landmarks):
        return ConvexHull(landmarks)

    def find_points_in_convexHull(self, points, hull):
        return np.array([self.check_if_point_in_convexHull(point, hull)
                         for point in points])

    def check_if_point_in_convexHull(self, point, hull):
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= config['convexHull']['tol'])
            for eq in hull.equations)

    def visualize_convexHull(self, level, digits, landmark_numbers):
        landmark_indices = np.array(self.get_landmark_indices(level, digits))[landmark_numbers]
        voltage_embedding = self.get_voltage_embedding(level, digits)
        mds_emb, nlm = self.multi_dim_scaling(voltage_embedding)
        mds_emb = mds_emb[:, 0:2]

        # voltage_embedding = np.flip(np.sort(voltage_embedding, axis=1), axis=1)
        # voltage_embedding = voltage_embedding[:, 0:len(landmark_numbers)-1]

        #lm_v_emb = voltage_embedding[landmark_indices, :]
        #lm_v_emb = np.flip(np.sort(lm_v_emb, axis=1), axis=1)
        #lm_v_emb = lm_v_emb[:, 0:3]
        #hull = self.find_convexHull(lm_v_emb)
        #is_in_hull = self.find_points_in_convexHull(voltage_embedding, hull)
        lm_mds_emb = mds_emb[landmark_indices, :]
        hull = self.find_convexHull(lm_mds_emb)
        is_in_hull = self.find_points_in_convexHull(mds_emb, hull)

        #mds_emb, nlm = self.multi_dim_scaling(voltage_embedding)
        #lm_mds_emb = mds_emb[landmark_indices, :]
        points_in_hull_mds_emb = mds_emb[is_in_hull, :]
        points_outside_hull_mds_emb = mds_emb[~is_in_hull, :]

        self.plotting_points_in_convexHull(lm_mds_emb, points_in_hull_mds_emb, points_outside_hull_mds_emb, hull)


    def plotting_points_in_convexHull(self, landmarks, points_in_hull, points_not_in_hull, hull):

        # We plot the mds embedding in 2D so we must remove higher dimensions
        landmarks = landmarks[:, 0:2]
        points_in_hull = points_in_hull[:, 0:2]
        points_not_in_hull = points_not_in_hull[:, 0:2]

        for simplex in hull.simplices:
            plt.plot(landmarks[simplex, 0], landmarks[simplex, 1])

        plt.scatter(*landmarks.T, alpha=.5, color='k', s=200, marker='v')
        plt.scatter(points_in_hull[:, 0], points_in_hull[:, 1], marker='x', color='g')
        plt.scatter(points_not_in_hull[:, 0], points_not_in_hull[:, 1], marker='d', color='m')

    def get_voltage_embedding(self, level, digits):
        voltage_embedding = []
        for digit in digits:
            voltage_embedding += self.voltages_dict[f'level{level}'][f'digit{digit}']
        return np.array(voltage_embedding).transpose()

    def multi_dim_scaling(self, voltage_embedding):
        return multi_dim_scaling(voltage_embedding), voltage_embedding.shape[1]

    def visualize_mds_embedding(self, level, digits, experiment_name):
        voltage_embedding = self.get_voltage_embedding(level, digits)
        mds_emb, nlm = self.multi_dim_scaling(voltage_embedding)
        source_indices = self.get_source_indices(level, digits)
        landmark_indices = self.get_landmark_indices(level, digits)
        self.plot_images_xy_plane(mds_emb, landmark_indices, source_indices, level, digits, nlm, experiment_name)

    def plot_images_xy_plane(self, data, labels, mds_emb, idx_lms, level, digits, nlm, experiment_name):
        cmap = color_map()
        cmap_mnist = color_map_for_mnist()

        fig = PLT.gcf()
        fig.clf()
        ax = PLT.subplot(111)

        plt.scatter(mds_emb[:, 0], mds_emb[:, 1], s=4, facecolor=cmap['color_darkgray'], lw=0)

        # Add images to the coordinates
        for idx in range(0, len(data)):
            digit = int(labels[idx])
            image = data[idx]
            image = image.reshape(28, 28)
            imagebox = OffsetImage(image, zoom=0.5, cmap=cmap_mnist[digit])
            xy = [mds_emb[idx, 0], mds_emb[idx, 1]]  # coordinates to position this image

            ab = AnnotationBbox(imagebox, xy,
                                xybox=(1., -1.),
                                xycoords='data',
                                frameon=False,
                                boxcoords="offset points")
            ax.add_artist(ab)

        # Add source landmarks
        for idx in idx_lms:
            digit = int(labels[idx])
            lm = data[idx]
            lm = lm.reshape(28, 28)
            imagebox = OffsetImage(lm, zoom=1.5, cmap=cmap_mnist[digit+2])
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

        ax.grid(True)
        PLT.draw()
        PLT.show()

        folder = self.config['folders']['results_folder']
        filepath = os.path.join(folder[0], folder[1] + '_' + f'{experiment_name}')
        Path(filepath).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(filepath, f'mnist_emb_lvl{level}_digits{digits}_nlm{nlm}.png'))
        fig.savefig(os.path.join(filepath, f'mnist_emb_lvl{level}_digits{digits}_nlm{nlm}.eps'), format='eps')

    def load_mnist_embedding(self, experiment_nr):
        folder = self.config['folders']['results_folder']
        filepath = os.path.join(folder[0], folder[1] + '_' + f'expnr{experiment_nr}')
        Path(filepath).mkdir(parents=True, exist_ok=True)

        with np.load(os.path.join(filepath, 'mnist_data.npz')) as data:
            mnist_data = data['mnist_data']
            mnist_labels = data['mnist_labels']

        filename = os.path.join(filepath, 'landmark_indices.json')
        with open(filename, 'r') as file:
            landmark_indices_dict = json.load(file)

        filename = os.path.join(filepath, 'source_indices.json')
        with open(filename, 'r') as file:
            source_indices_dict = json.load(file)

        filename = os.path.join(filepath, 'voltages.json')
        with open(filename, 'r') as file:
            voltages_dict = json.load(file)

        with open(os.path.join(filepath, 'digit_indices')) as csv_file:
            csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
            digit_indices = []
            for row in csv_reader:
                row = [int(e) for e in row]  # Convert to int
                digit_indices.append(row)
        digit_indices = np.array(digit_indices)

        return mnist_data, mnist_labels, digit_indices, landmark_indices_dict, source_indices_dict, voltages_dict


if __name__ == '__main__':
    load_experiment_nr = 2
    mnist_embedding = MNIST_embedding_visualization(config, experiment_nr=load_experiment_nr)

    # Visualize the multi-dim scaling embedding
    #level = 0
    #digits = config['mnist']['digits']
    #visualization_exp_name = 'mds_emb_nr1'
    #mnist_embedding.visualize_mds_embedding(level, digits, experiment_name=visualization_exp_name)

    level = 0
    digits = config['mnist']['digits']
    #landmark_numbers = [5,6,7,8] # This is a good subset of landmarks for digit 3 when 20 total lm
    landmark_numbers = [13, 15, 16, 17]  # This is an interesting subset for digit 4 when 20 total lm
    visualization_exp_name = 'convexhull_nr1'
    #mnist_embedding.visualize_convexHull(level, digits, landmark_numbers)
    mnist_embedding.visualize_voltageSpace_convexHull(level, digits, landmark_numbers)
    plt.show()