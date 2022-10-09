from utilities.mnist_exp_util import internal_distances
from utilities.mnist_exp_util import load_mnist, pre_processing

import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    datasize = 500

    mnistdata, target = load_mnist()
    mnistdata, target = pre_processing(datasize, mnistdata, target)

    # Check internal distances, to
    internal_distances(mnistdata)
    plt.show()