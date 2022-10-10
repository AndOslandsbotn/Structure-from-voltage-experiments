## mnist_voltage_embedding.py file

#### MNIST_Embedding

This class generates an embedding of the MNIST dataset

   Variables
   - mnist_data: The mnist images as 784 dimensional vectors, stored as n x 784 numpy matrix
   - labels: The labels of each image, numpy array of length n
   - digit_indices:  List of length 10, where 
   element nr i is a numpy array containing indices 
   to digit nr i in the mnist dataset
   - landmark_indices_dict: Nested dictionary containing all landmarks organized according to level and digit
   - source_indices_dict: Nested dictionary containing all source indices organized according to level and digit
   - voltages_dict  # Nested dictionary containing voltage functions from each landmark, organized according to level and digit
   - levels: a set containing the levels
   - rhoG_dict: a dictionary containing the rhoG of all landmark that has rhoG different from default


    ## Public methods:
    # add_landmarks: Method that adds landmarks to a specific level and digit type
    # calc_voltages: Calculates the voltages at all the landmarks that have been added
    # plot_voltage_decay: Plot the voltage decay and save the figure to the results folder
    # get_data: Returns the mnist data, labels and digit_indices
    # specify_rhoG: Method that allows for specifying a rhoG for a particular landmark (inverse of resistance to ground) that is different from the
        default value

    ## Private methods:
    # find_source_indices: Get indices of all points in x that are distance $r_s$ from the landmark
    # calc_voltage: Calculates the voltage associated to a specific landmark
    # save_experiment
    """