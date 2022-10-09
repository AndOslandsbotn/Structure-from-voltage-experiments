import numpy as np
import os
import json
import csv
from pathlib import Path
from scipy.spatial.distance import cdist

from utilities.visualization import find_convexHull, plotting_points_in_convexHull
from utilities.util import flatten_list
from utilities.multidimensional_scaling import multi_dim_scaling
from utilities.color_maps import color_map, color_map_for_mnist, color_map_list

from definitions import CONFIG_VIZ_PATH
from config.yaml_functions import yaml_loader
config = yaml_loader(CONFIG_VIZ_PATH)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import Rectangle, gca
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

class MNIST_Visualization_Base():
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

    def construct_voltage_embedding(self, level, digits):
        voltage_embedding = []
        for digit in digits:
            voltage_embedding += self.voltages_dict[f'level{level}'][f'digit{digit}']
        return np.array(voltage_embedding).transpose()[0:-1, :]  # We do [0:-1] to remove ground node

    def construct_mds_embedding(self, voltage_embedding):
        return multi_dim_scaling(voltage_embedding)

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

    def plot_images_xy_plane(self, data, labels, mds_emb, idx_lms, level, digits, nlm, experiment_name):
        cmap = color_map()
        cmap_mnist = color_map_for_mnist()

        fig = plt.gcf()
        fig.clf()
        ax = plt.subplot(111)

        # Plot the multi-dim scaling embedding in 2D
        plt.scatter(mds_emb[:, 0], mds_emb[:, 1], s=4, facecolor=cmap['color_darkgray'], lw=0)

        # Add images to the coordinates provided by the multi-dim scaling embedding
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
            ab.set(zorder=1)

        # Add source landmarks, represent them with extra large digits
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
            ab.set(zorder=3)

        plt.xticks(color='w')
        plt.yticks(color='w')
        ax.set_facecolor('black')
        ax.add_artist(ab)

        ax.grid(True)
        return fig, ax

    def save_figure(self, fig, ax, filepath, filename):
        ax.set_rasterization_zorder(1)
        Path(filepath).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(filepath, filename))
        fig.savefig(os.path.join(filepath, filename + '.pdf'), format='pdf')
        fig.savefig(os.path.join(filepath, filename + '.eps'), format='eps')

from utilities.util import normalize_voltage
class MNIST_Visualization_Global(MNIST_Visualization_Base):
    def __init__(self, config, experiment_nr):
        super().__init__(config, experiment_nr)

    def visualize_mds_embedding(self, level, digits, experiment_name):
        voltage_embedding = self.construct_voltage_embedding(level, digits)
        voltage_embedding = normalize_voltage(voltage_embedding)

        mds_emb = self.construct_mds_embedding(voltage_embedding)
        nlm = voltage_embedding.shape[1]
        landmark_indices = self.get_landmark_indices(level, digits)
        fig, ax = self.plot_images_xy_plane(self.data, self.labels, mds_emb, landmark_indices, level, digits, nlm, experiment_name)
        plt.show()

        # Save figure to file
        folder = self.config['folders']['results_folder']
        filepath = os.path.join(folder[0], folder[1] + '_' + f'{experiment_name}')
        filename = f'{experiment_name}_level{level}_digits{digits}_nlm{nlm}'
        self.save_figure(fig, ax, filepath, filename)

class MNIST_Visualization_Local(MNIST_Visualization_Base):
    def __init__(self, config, experiment_nr):
        super().__init__(config, experiment_nr)

    def calc_mutual_distances(self, x):
        return cdist(x, x, self.config['local_embedding']['metric'])

    def find_max_distances(self, landmarks):
        return np.max(self.calc_mutual_distances(landmarks), axis=1)

    def find_neighbourhood(self, landmark_indices, voltage_embedding):
        max_distances_landmarks = self.find_max_distances(voltage_embedding[landmark_indices, :])
        distances = self.calc_mutual_distances(voltage_embedding)
        distances_to_landmarks = distances[:, landmark_indices]

        condition = distances_to_landmarks > max_distances_landmarks + self.config['convexHull']['tol']
        is_neighbourhood = np.sum(condition, axis=1) == 0
        indices = np.arange(0, voltage_embedding.shape[0])
        return indices[is_neighbourhood]

    def visualize_mds_embedding(self, level, digits, desired_landmarks, experiment_name):
        landmark_indices = np.array(self.get_landmark_indices(level, digits))[desired_landmarks]
        voltage_embedding = self.construct_voltage_embedding(level, digits) #[0:-1, :]
        voltage_embedding = normalize_voltage(voltage_embedding)


        nlm = voltage_embedding.shape[1]

        # Find neighbourhood of desired landmarks
        indices_neighbourhood = self.find_neighbourhood(landmark_indices, voltage_embedding)
        voltage_embedding_neighbourhood = voltage_embedding[indices_neighbourhood, :]

        # Multi-dimensional scaling embedding on the neighbourhood
        mds_embedding_neighbourhood = self.construct_mds_embedding(voltage_embedding_neighbourhood)
        mnistdata_neighbourhood = self.data[indices_neighbourhood, :]
        labels_neighbourhood = self.labels[indices_neighbourhood]
        landmark_indices_neighbourhood = np.where(np.in1d(indices_neighbourhood, landmark_indices))[0]

        fig, ax = self.plot_images_xy_plane(mnistdata_neighbourhood,
                                  labels_neighbourhood,
                                  mds_embedding_neighbourhood,
                                  landmark_indices_neighbourhood,
                                  level, digits, nlm, experiment_name=experiment_name)
        plt.show()
        # Save figure to file
        folder = self.config['folders']['results_folder']
        filepath = os.path.join(folder[0], folder[1] + '_' + f'{experiment_name}')
        filename = f'local_mnist_emb_lvl{level}_digits{digits}_nlm{nlm}_desired_lm{desired_landmarks}'
        self.save_figure(fig, ax, filepath, filename)

    def highlight_local_embedding_region_on_global(self, ax, mds_embedding_neighbourhood, mnistdata_neighbourhood, labels_neighbourhood):
        cmap_mnist = color_map_for_mnist()

        # Add images to the coordinates provided by the multi-dim scaling embedding
        for idx in range(0, len(mds_embedding_neighbourhood)):
            digit = int(labels_neighbourhood[idx])
            image = mnistdata_neighbourhood[idx]
            image = image.reshape(28, 28)
            imagebox = OffsetImage(image, zoom=0.5, cmap=cmap_mnist[10])
            xy = [mds_embedding_neighbourhood[idx, 0], mds_embedding_neighbourhood[idx, 1]]  # coordinates to position this image

            ab = AnnotationBbox(imagebox, xy,
                                xybox=(1., -1.),
                                xycoords='data',
                                frameon=False,
                                boxcoords="offset points")
            ab.set(zorder=2)
            ax.add_artist(ab)

    def visualize_local_mds_embedding_in_global(self, level, digits, desired_landmarks, experiment_name):
        landmark_indices = np.array(self.get_landmark_indices(level, digits))[desired_landmarks]
        voltage_embedding = self.construct_voltage_embedding(level, digits)
        voltage_embedding = normalize_voltage(voltage_embedding)

        lm_v_emb = voltage_embedding[landmark_indices, :]
        nlm = voltage_embedding.shape[1]

        # Find local neighbourhood around desired landmarks
        indices_neighbourhood = self.find_neighbourhood(landmark_indices, voltage_embedding)

        # Construct global multi-dimensional scaling embedding
        mds_emb = self.construct_mds_embedding(voltage_embedding)
        #lm_mds_emb = mds_emb[landmark_indices, :]
        mds_emb_neighbourhood = mds_emb[indices_neighbourhood, :]
        mnistdata_neighbourhood = self.data[indices_neighbourhood, :]
        labels_neighbourhood = self.labels[indices_neighbourhood]

        # Find points that are inside neighbourhood and draw a convex hull in mds space
        #points_in_hull_mds_emb = mds_emb[indices_neighbourhood, :]
        #points_outside_hull_mds_emb = mds_emb[~indices_neighbourhood, :]
        #hull = find_convexHull(lm_mds_emb[:, 0:2])
        #plotting_points_in_convexHull(lm_mds_emb, points_in_hull_mds_emb, points_outside_hull_mds_emb, hull)

        fig, ax = self.plot_images_xy_plane(self.data,
                                  self.labels,
                                  mds_emb,
                                  landmark_indices,
                                  level, digits, nlm, experiment_name)
        self.highlight_local_embedding_region_on_global(ax, mds_emb_neighbourhood,
                                                        mnistdata_neighbourhood,
                                                        labels_neighbourhood
                                                        )

        plt.show()
        # Save figure to file
        folder = self.config['folders']['results_folder']
        filepath = os.path.join(folder[0], folder[1] + '_' + f'{experiment_name}')
        filename = f'local_vs_global_digits{digits}_nlm{nlm}_desired_lm{desired_landmarks}'
        self.save_figure(fig, ax, filepath, filename)

if __name__ == '__main__':
    level = 0
    digits = config['mnist']['digits']
    load_experiment_nr = 2
    embedding_type ='Local'

    if embedding_type == 'Global':
        visualization_exp_name = f'global_expnr{load_experiment_nr}'
        mnist_embedding_global = MNIST_Visualization_Global(config, experiment_nr=load_experiment_nr)
        mnist_embedding_global.visualize_mds_embedding(level, digits, experiment_name=visualization_exp_name)

    elif embedding_type == 'Local':
        mnist_embedding_local = MNIST_Visualization_Local(config, experiment_nr=load_experiment_nr)
        #desired_landmarks = [1, 2, 3]  # For digit 3 when 10 total lm
        #desired_landmarks = [5,6,7] # For digit 4 when 10 total lm
        desired_landmarks = [5, 6, 7, 8] # This is a good subset of landmarks for digit 3 when 20 total lm
        #desired_landmarks = [13, 15, 16, 17]  # This is an interesting subset for digit 4 when 20 total lm

        visualization_exp_name = f'local_expnr{load_experiment_nr}'
        mnist_embedding_local.visualize_mds_embedding(level,
                                                      digits,
                                                      desired_landmarks,
                                                      experiment_name=visualization_exp_name)

        visualization_exp_name = f'local_expnr{load_experiment_nr}'
        mnist_embedding_local.visualize_local_mds_embedding_in_global(level,
                                                                      digits,
                                                                      desired_landmarks,
                                                                      experiment_name=visualization_exp_name)