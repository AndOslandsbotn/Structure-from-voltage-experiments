import scipy.spatial as spatial
import numpy as np

def get_union(list1, list2):
    """Get union of two lists"""
    return list(set(list1 + list2))

def get_intersection(list1, list2):
    """Get intersetction between two lists"""
    set1 = set(list1)
    intersect = set1.intersection(set(list2))
    return list(intersect)

def get_nn_indices(x, y, r):
    """Return indices for all points in x that ar within radius r from y
    :param x: n x d numpy vector of n examples of dimension d
    :param y: 1 x d numpy vector
    :param r: cut-of radius
    """
    tree = spatial.KDTree(x)
    indices = tree.query_ball_point(y, r)
    nn_indices = {i: set(idx) for i, idx in enumerate(indices)}
    nn_vector = [item for sublist in indices for item in sublist]
    return nn_indices, nn_vector


def flatten_list(l):
    """Takes in a list of lists and flattens the outer list to one list"""
    return [item for sublist in l for item in sublist]

def normalize_voltage(voltages):
    """Normalize the voltage functions to have max value 1"""
    max_voltage = np.max(voltages, axis=0)
    return voltages/max_voltage
