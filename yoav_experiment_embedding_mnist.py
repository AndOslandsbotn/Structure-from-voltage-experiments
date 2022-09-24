import numpy as np
import matplotlib.pyplot as plt

from utilities.voltage_solver import propagate_voltage, apply_voltage_constraints
from utilities.mnist_exp_util import load_mnist, pre_processing, organize_digits
from utilities.matrices import construct_W_matrix
from utilities.util import get_nn_indices

if __name__ == '__main__':
    config = {
        'kernelType': 'radial_scaled',
        'max_iter': 1000,
        'is_Wtilde': False
    }

    # Variables
    bw = 8  # Bandwidth
    rhoG = 1.e-3  # Inverse of resistance to ground
    rs = 8  # Source radius

    num_lm_per_digit = 1
    nlm = 10 * num_lm_per_digit
    datasize = 500  # Size of mnistdata that we look at
    ######################

    mnistdata, target = load_mnist()
    mnistdata, target = pre_processing(datasize, mnistdata, target)

    digit_indices = organize_digits(target)

    # Choose a landmark
    digit_type = 1  # Choose a landmark of digit 1
    random_idx = np.random.choice(digit_indices[digit_type], size=1)
    landmark = mnistdata[random_idx]

    # This is how the selected landmark looks like
    #plt.figure()
    #plt.imshow(lm.reshape(28, 28))
    #plt.show()

    # Construct the adjacency matrix W
    matrix = construct_W_matrix(mnistdata, datasize, bw, rhoG, config)

    # Get indices of all points in x that are distance $r_s$ from the landmark
    source_indices, _ = get_nn_indices(mnistdata, landmark.reshape(1, -1), rs)
    source_indices = list(source_indices[0])

    # Initialize a voltage vector, with source and ground constraints applied
    init_voltage = np.zeros(datasize + 1)
    init_voltage = apply_voltage_constraints(init_voltage, source_indices)

    # Propagate the voltage to all points in the dataset
    voltage = propagate_voltage(init_voltage, matrix, config['max_iter'],
                                          source_indices, config['is_Wtilde'],
                                          is_visualization=False)

    plt.figure()
    plt.plot(np.sort(voltage))

    plt.figure()
    plt.plot(np.sort(voltage), label=f'Landmark of digit {digit_type}')
    plt.xlabel('sample points sorted after smallest to largest voltage')
    plt.ylabel('Voltage')
    plt.yscale('log')
    plt.show()