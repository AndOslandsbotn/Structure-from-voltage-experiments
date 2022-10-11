from mnist.mnist_util import load_mnist, pre_processing
from utilities.util import get_nn_indices
from utilities.matrices import construct_wtilde_matrix, construct_w_matrix
from utilities.voltage_solver import propagate_voltage, apply_voltage_constraints
import numpy as np
import json
from pathlib import Path
import os
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

from definitions import CONFIG_PATH, ROOT_DIR
from config.yaml_functions import yaml_loader
config = yaml_loader(CONFIG_PATH)

if __name__ == '__main__':
    print(json.dumps(config, indent=4, sort_keys=True))

np.random.seed(42)

class MNIST_Embedding():
    """This class generates an embedding of the MNIST dataset

    ## Variables:
    # mnist_data   # The mnist images as 784 dimensional vectors, stored as n x 784 numpy matrix
    # labels  # The labels of each image, numpy array of length n
    # digit_indices  # List of length 10, where element nr i is a numpy array
    # containing indices to digit nr i in the mnist dataset
    # landmark_indices_dict  # Nested dictionary containing all landmarks organized according to level and digit
    # source_indices_dict  # Nested dictionary containing all source indices organized according to level and digit
    # voltages_dict  # Nested dictionary containing voltage functions from each landmark, organized according to level and digit
    # levels  # a set containing the levels
    # rhoG_dict  # a dictionary containing the rhoG of all landmark that has rhoG different from default

    ## Public methods:
    # add_landmarks
    # calc_voltages
    # plot_voltage_decay
    # get_data
    # specify_rhoG

    ## Private methods:
    # find_source_indices
    # calc_voltage
    # save_experiment
    """

    def __init__(self, config):
        self.config = config
        mnist_data, mnist_labels = load_mnist()
        results = pre_processing(mnist_data,
                                 mnist_labels,
                                 datasize=self.config['mnist']['datasize'],
                                 digit_types=self.config['mnist']['digits'])
        self.mnist_data, self.mnist_labels, self.datasize, self.digit_indices = results

        self.digits = self.config['mnist']['digits']
        self.landmark_indices_dict = {}
        self.source_indices_dict = {}
        self.voltages_dict = {}
        self.levels = set()

        self.rhoG_dict = {}  # Specific rhoGs can be added to this dictionary for particular landmarks

    def get_data(self):
        """Returns the mnist data, labels and digit_indices"""
        return self.mnist_data, self.mnist_labels, self.digit_indices

    def specify_rhoG(self, level, digit, landmark_nr, rhoG):
        """Method that allows for specifying a rhoG for a particular landmark (inverse of resistance to ground) that is different from the
        default value
        :param level: Level, this is relevant if we want to do multi-resolution
        :param digit: Digit between 0 and 10
        :param landmark_nr: The number of the landmarks
        :param rhoG: The inverse of the resistance to ground, that we want to assign to the landmark
        """
        if not f'level{level}' in self.rhoG_dict.keys():
            self.rhoG_dict[f'level{level}'] = {}
        if not f'digit{digit}' in self.rhoG_dict[f'level{level}'].keys():
            self.rhoG_dict[f'level{level}'][f'digit{digit}'] = {}
        self.rhoG_dict[f'level{level}'][f'digit{digit}'][landmark_nr] = rhoG

    def add_landmarks(self, landmark_indices, level, digit_type):
        """Method that adds landmarks to a specific level and digit type
        :param landmark_indices: index of the landmarks that we are adding
        :param level: level that we add the landmarks to
        :param digit_type: digit type(s) of the landmarks
        """
        if not level in self.levels:
            self.levels.add(level)
            self.landmark_indices_dict[f'level{level}'] = {}
            self.source_indices_dict[f'level{level}'] = {}
            self.voltages_dict[f'level{level}'] = {}
        self.landmark_indices_dict[f'level{level}'][f'digit{digit_type}'] = landmark_indices.tolist()
        self.find_source_indices(landmark_indices, level, digit_type)

    def find_source_indices(self, landmark_indices, level, digit_type):
        """Get indices of all points in x that are distance $r_s$ from the landmark
        :param landmark_indices: index of the landmarks that we are adding
        :param level: level that we add the landmarks to
        :param digit_type: digit type(s) of the landmarks
        """
        source_idxs_l = []
        for landmark_idx in landmark_indices:
            landmark = self.mnist_data[landmark_idx]
            source_idxs, _ = get_nn_indices(self.mnist_data, landmark.reshape(1, -1), config['metricGraph']['source_radius'])
            source_idxs_l.append(list(source_idxs[0]))
        self.source_indices_dict[f'level{level}'][f'digit{digit_type}'] = source_idxs_l

    def calc_voltages(self, experiment_nr):
        """ Calculates the voltages at all the landmarks that have been added
        :param experiment_nr: label to store the experiment
        """
        for level in tqdm(self.levels, desc=f'iterate levels: {self.levels}'):
            for digit in tqdm(self.digits, desc=f'iterate digits: {self.digits}'):
                voltages_l = []
                for landmark_nr, source_indices in enumerate(self.source_indices_dict[f'level{level}'][f'digit{digit}']):
                    try: # landmark_nr in self.rhoG_dict[f'level{level}'][f'digit{digit}'].keys():
                        rhoG = self.rhoG_dict[f'level{level}'][f'digit{digit}'][landmark_nr]
                        print(f"We modify lvl{level}, digit{digit}, landmark nr{landmark_nr}")
                    except KeyError:
                        rhoG = self.config['metricGraph']['rhoG']
                        print(f"We Use default rhoG, lvl{level}, digit{digit}, landmark nr{landmark_nr}")
                    voltages_l.append(self.calc_voltage(source_indices, rhoG).tolist())
                self.voltages_dict[f'level{level}'][f'digit{digit}'] = voltages_l
        self.plot_voltage_decay(experiment_nr)
        self.save_experiment(experiment_nr)

    def calc_voltage(self, source_indices, rhoG):
        """ Calculates the voltage associated to a specific landmark
        :param source_indices: Indices to all nodes that are radius r_s from a specific landmark
        :param rhoG: the inverse resistance to ground associated with landmark
        """

        # Construct the modulated adjacency matrix W-tilde
        weight_matrix = construct_w_matrix(self.mnist_data, rhoG, config)

        # Initialize a voltage vector, with source and ground constraints applied
        init_voltage = np.zeros(self.datasize + 1)
        init_voltage = apply_voltage_constraints(init_voltage, source_indices)

        # Propagate the voltage to all points in the dataset
        return propagate_voltage(init_voltage, weight_matrix, source_indices, config)

    def plot_voltage_decay(self, experiment_nr):
        """ Plot the voltage decay and save the figure to the results folder
        :param experiment_nr: label to store the experiment
        """
        folder = self.config['folders']['results_folder']
        filepath = os.path.join(folder[0], folder[1] + '_' + f'expnr{experiment_nr}')
        Path(filepath).mkdir(parents=True, exist_ok=True)

        for level in self.levels:
            for digit in self.digits:
                plt.figure()
                for i, voltage in enumerate(self.voltages_dict[f'level{level}'][f'digit{digit}']):
                    voltage_sort = np.sort(voltage, axis=0)
                    plt.plot(voltage_sort, label=f'Landmark nr {i}, Digit {digit}')
                plt.legend()
                plt.savefig(os.path.join(filepath, f'VoltageDecayMnist_digit{digit}'))
        plt.show()

    def save_experiment(self, experiment_nr):
        """Saves the voltage curves to a file specifid by the results_folder in the config file
        :param experiment_nr: label to store the experiment
        """
        print("Save experiment")

        folder = self.config['folders']['results_folder']
        filepath = os.path.join(folder[0], folder[1]+ '_'+f'expnr{experiment_nr}')
        Path(filepath).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(filepath, 'landmark_indices.json')
        with open(filename, 'w') as writefile:
            json.dump(self.landmark_indices_dict, writefile)

        filename = os.path.join(filepath, 'source_indices.json')
        with open(filename, 'w') as writefile:
            json.dump(self.source_indices_dict, writefile)

        filename = os.path.join(filepath, 'voltages.json')
        with open(filename, 'w') as writefile:
            json.dump(self.voltages_dict, writefile)

        with open(os.path.join(filepath, 'digit_indices'), 'w', newline="") as f:
            write = csv.writer(f)
            write.writerows(self.digit_indices)

        filename = os.path.join(filepath, 'config_used.json')
        with open(filename, 'w') as writefile:
            json.dump(self.config, writefile)

        filename = os.path.join(filepath, 'mnist_data.npz')
        np.savez(filename, mnist_data=self.mnist_data, mnist_labels=self.mnist_labels,)
        return

