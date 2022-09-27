import numpy as np
import matplotlib.pyplot as plt

from utilities.voltage_solver import propagate_voltage, apply_voltage_constraints
from utilities.mnist_exp_util import load_mnist, pre_processing, organize_digits
from utilities.matrices import construct_W_matrix
from utilities.util import get_nn_indices

class mnist:
    def __init__(self,datasize,digit_type=4):
        mnistdata, target = load_mnist()
        print(mnistdata.shape)
        #get all of the images for one digit.
        self.mnistdata, self.target, self.datasize = pre_processing(0, mnistdata, target,digit_type=digit_type)
        self.config = {
            'kernelType': 'radial_scaled',
            'max_iter': 100,
            'is_Wtilde': False
        }
        
    def calc_voltages(self,bw=8, rhoG=1.e-3, rs=8,digit_type=1, num_landmarks=10):
        """ calculate voltage functions for randomply chosen landmarks of a given digit 
        bw      Bandwidth
        rhoG    Inverse of resistance to ground
        rs      Source radius
        digit_type          the type of digit (0-9)
        num_landmarks       number of landmarks.
        """

        digit_indices = organize_digits(self.target)

        # Choose a landmark
        random_idx = np.random.choice(digit_indices[digit_type], size=num_landmarks)

        voltages=[]
        source_indices_list=[]
        labels=[]
        j=0
        total=len(random_idx)
        for landmark_index in random_idx:
            print('\r',j,'/',total,end='')
            j+=1
            
            labels.append(self.target[landmark_index])

            landmark = self.mnistdata[landmark_index]

            # Construct the adjacency matrix W
            matrix = construct_W_matrix(self.mnistdata, self.datasize, bw, rhoG, self.config)

            # Get indices of all points in x that are distance $r_s$ from the landmark
            source_indices, _ = get_nn_indices(self.mnistdata, landmark.reshape(1, -1), rs)
            source_indices = list(source_indices[0])

            # Initialize a voltage vector, with source and ground constraints applied
            init_voltage = np.zeros(self.datasize + 1)
            init_voltage = apply_voltage_constraints(init_voltage, source_indices)

            # Propagate the voltage to all points in the dataset
            voltage = propagate_voltage(init_voltage, matrix, self.config['max_iter'],
                                                  source_indices, self.config['is_Wtilde'],
                                                  is_visualization=False)

            #normalize voltages so that current out of source and into ground is 1.
            I=np.sum(voltage)*rhoG
            voltage=voltage/I

            voltages.append(voltage)
            source_indices_list.append(source_indices)

        return np.stack(voltages),random_idx,source_indices_list,labels

    def plot_landmark(self,indices,num_col=5,title=None):
        num_row=int(len(indices)/num_col)+1
        plt.figure(figsize=[num_col*2,num_row*2])
        if not title is None:
            plt.title(title)
        for i in range(len(indices)):
            plt.subplot(num_row,num_col,i+1)
            index=indices[i]
            plt.imshow(self.mnistdata[index,:].reshape(28, 28))
            plt.title(str(self.target[index]))
        plt.show()

if __name__ == '__main__':

    MN=mnist(datasize=500)
    V,lm_idx,source_indices_list,labels=MN.calc_voltages(num_landmarks=2,digit_type=4)
    print(V.shape,lm_idx,len(source_indices_list),len(labels))

    MN.plot_landmark(lm_idx)
    
    # plt.figure()
    # plt.plot(np.sort(voltage))

    # plt.figure()
    # plt.plot(np.sort(voltage), label=f'Landmark of digit {digit_type}')
    # plt.xlabel('sample points sorted after smallest to largest voltage')
    # plt.ylabel('Voltage')
    # plt.yscale('log')
    # plt.show()


    def calc_ER(V1,s1,V2,s2):
    selfV1=np.mean(V1[s1])
    selfV2=np.mean(V2[s2])
    cross12=np.mean(V1[s2])
    cross21=np.mean(V2[s1])
    dist=selfV1+selfV2-cross12-cross21
    return dist,selfV1,selfV2,cross12,cross21
