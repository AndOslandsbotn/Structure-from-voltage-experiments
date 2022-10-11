from mnist.mnist_voltage_embedding import MNIST_Embedding
import numpy as np
import json

from definitions import CONFIG_PATH
from config.yaml_functions import yaml_loader
config = yaml_loader(CONFIG_PATH)

if __name__ == '__main__':
    print(json.dumps(config, indent=4, sort_keys=True))

np.random.seed(42)

if __name__ == '__main__':
    experiment_nr = 2
    mnist = MNIST_Embedding(config)
    mnist_data, labels, digit_indices = mnist.get_data()

    # Select landmarks digit 3, level 0
    digits = config['mnist']['digits']
    landmarks_per_digit = config['mnist']['landmarks_per_digit']
    for nlm, digit in zip(landmarks_per_digit, digits):
        landmark_indices = np.random.choice(digit_indices[digit], size=nlm).reshape(-1, 1)
        mnist.add_landmarks(landmark_indices, level=0, digit_type=digit)

    mnist.specify_rhoG(level=0, digit=3, landmark_nr=0, rhoG=1e-3)
    #mnist.specify_rhoG(level=0, digit=3, landmark_nr=4, rhoG=1e-2)

    mnist.calc_voltages(experiment_nr=experiment_nr)
