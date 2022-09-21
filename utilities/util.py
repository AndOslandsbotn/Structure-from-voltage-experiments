import scipy.spatial as spatial

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