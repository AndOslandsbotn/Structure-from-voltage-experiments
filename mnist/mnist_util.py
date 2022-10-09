import numpy as np
import os
from definitions import ROOT_DIR
from scipy.spatial import ConvexHull

def load_mnist():
    """Load mnist data"""
    with np.load(os.path.join(ROOT_DIR, 'Data', 'mnist.npz')) as data:
        x = data['x_train'].reshape(-1, 28*28)
        y = data['y_train']
    return x, y

def pre_processing(mnistdata, labels, datasize=-1, digit_types=-1):
    """Preprocessing of the mnist data. Select dataset of size datasize and normalize using
    max value in the selected dataset"""
    print('pre_processing', mnistdata.shape)
    assert all(digit >= 0 and digit <= 9 for digit in digit_types)

    # Wether to take the entire dataset
    if datasize < 0:
        datasize = mnistdata.shape[0]

    # We select the digits
    digit_indices = np.sum(np.array([labels == digit for digit in digit_types]), axis=0)
    mnistdata=mnistdata[digit_indices == True,:]
    labels=labels[digit_indices == True]

    # Reduce the data size
    idx = np.random.choice(np.arange(0, len(mnistdata)), size=min(mnistdata.shape[0], datasize))
    mnistdata = mnistdata[idx]
    target = labels[idx]

    # Normalize
    mnistdata = mnistdata/np.max(mnistdata)

    # Organize the digits
    digit_indices = organize_digits(target)
    return mnistdata, target, mnistdata.shape[0], digit_indices

def organize_digits(target):
    """This function finds the indices of all digit 1,2,3,4 etc.
    such that we know where each type of digit is.
    :param
    :returns digit_indices: This is a list of numpy arrays. numpy array nr i
    in the list contains indices to all points that is digit i.
    """
    target_names = np.arange(0, 10)
    digit_indices = []
    for digit_nr in target_names:
        indices = np.where(target == digit_nr)[0]
        digit_indices.append(indices)
    return digit_indices

