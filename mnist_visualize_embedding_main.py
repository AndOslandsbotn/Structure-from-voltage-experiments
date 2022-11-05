from mnist.mnist_visualize_embedding import MNIST_Visualization_Local, MNIST_Visualization_Global
from definitions import CONFIG_VIZ_PATH
from config.yaml_functions import yaml_loader
config = yaml_loader(CONFIG_VIZ_PATH)

if __name__ == '__main__':
    level = 0
    digits = config['mnist']['digits']
    load_experiment_nr = 3
    embedding_type ='Global'

    if embedding_type == 'Global':
        visualization_exp_name = f'expnr{load_experiment_nr}'
        mnist_embedding_global = MNIST_Visualization_Global(config, experiment_nr=load_experiment_nr)
        mnist_embedding_global.visualize_mds_embedding(level, digits, experiment_name=visualization_exp_name)

    elif embedding_type == 'Local':
        mnist_embedding_local = MNIST_Visualization_Local(config, experiment_nr=load_experiment_nr)
        #desired_landmarks = [1, 2, 3]  # For digit 3 when 10 total lm
        desired_landmarks = [5,6,7] # For digit 4 when 10 total lm
        #desired_landmarks = [5, 6, 7, 8] # This is a good subset of landmarks for digit 3 when 20 total lm
        #desired_landmarks = [13, 15, 16, 17]  # This is an interesting subset for digit 4 when 20 total lm

        visualization_exp_name = f'expnr{load_experiment_nr}'
        mnist_embedding_local.visualize_mds_embedding(level,
                                                      digits,
                                                      desired_landmarks,
                                                      experiment_name=visualization_exp_name)

        visualization_exp_name = f'expnr{load_experiment_nr}'
        mnist_embedding_local.visualize_local_mds_embedding_in_global(level,
                                                                      digits,
                                                                      desired_landmarks,
                                                                      experiment_name=visualization_exp_name)