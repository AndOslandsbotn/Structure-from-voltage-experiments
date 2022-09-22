import numpy as np
import math as m
import os

import matplotlib
import matplotlib.pyplot as plt

def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])

def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])

def Rz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta), m.cos(theta), 0],
                      [0, 0, 1]])

def sample_2d_unit_square(n, eps):
    return np.random.uniform(low=0, high=1, size=(n, 2))

def sample_data_uniform_sphere(N, D, sigma_eps):
    # Sample detachment coefficients from ||k||_2 < noise_level * sqrt(D - 1)
    X = np.random.normal(size = (D, N))
    X = X/np.linalg.norm(X, axis = 0)
    X = X.T
    return X

