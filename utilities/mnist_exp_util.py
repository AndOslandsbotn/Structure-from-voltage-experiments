from scipy.spatial.distance import cdist
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os
import csv

def internal_distances(mnistdata):
    distances = cdist(mnistdata, mnistdata)
    plt.figure()
    indices = np.random.choice(np.arange(0, len(mnistdata)), size=100)
    for idx in indices:
        plt.plot(np.sort(distances[idx, :]))

def load_mnist():
    with np.load(os.path.join('Data', 'mnist.npz')) as data:
        x = data['x_train'].reshape(-1, 28*28)
        y = data['y_train']
    return x, y

def pre_processing(datasize, mnistdata, target):
    idx = np.random.choice(np.arange(0, len(mnistdata)), size=datasize)
    mnistdata = mnistdata[idx]
    target = target[idx]
    mnistdata = mnistdata/np.max(mnistdata)
    return mnistdata, target


def organize_digits(target):
    """This function finds the indices of all digit 1,2,3,4 such that we know where each type of digit is
    """
    target_names = np.arange(0, 10)
    digit_indices = []
    for digit_nr in target_names:
        indices = np.where(target == digit_nr)[0]
        digit_indices.append(indices)
    return digit_indices

def select_landmarks_mnist_standard(data, digit_indices, num_lm_per_digit):
    """Standard method for selecting landmarks from mnist. Selects 'num_lm_per_digit'
    randomly for each of the 10 digits"""
    landmarks = []
    idx_lms = []
    for digit_idx in digit_indices:
        indices = np.random.choice(digit_idx, size=min(num_lm_per_digit, len(digit_idx)))
        for idx in indices:
            idx = np.array([idx])
            idx_lms.append(idx)
            landmarks.append(data[idx])
    return landmarks, idx_lms
