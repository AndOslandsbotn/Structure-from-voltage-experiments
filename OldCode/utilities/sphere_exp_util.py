import math
import numpy as np

from utilities.util import get_intersection

def cart2polar(X):
    R = np.sqrt(np.sum(np.square(X), axis=1))
    phi = np.arctan2(X[:, 1], X[:, 0])
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    theta = np.arctan2(r, X[:, 2])
    return R, theta, phi

def polar2cart(R, theta, phi):
    #theta = np.pi / 180 * theta
    #phi = np.pi / 180 * phi

    phi = math.radians(phi)
    theta = math.radians(theta)
    x = R*np.sin(theta) * np.cos(phi)
    y = R*np.sin(theta) * np.sin(phi)
    z = R*np.cos(theta)
    return x, y, z

def get_sphere_section(x, limits):
    theta_min = math.radians(limits['theta_min'])
    theta_max = math.radians(limits['theta_max'])
    phi_min = math.radians(limits['phi_min'])
    phi_max = math.radians(limits['phi_max'])

    R, theta, phi = cart2polar(x)
    idx2 = (theta >= theta_min) & (theta <= theta_max)
    idx2 = np.where(idx2 == True)[0]
    idx3 = (phi >= phi_min) & (phi <= phi_max)
    idx3 = np.where(idx3 == True)[0]

    # only consider positive y values
    idx1 = np.where(x[:, 1] > 0)[0]
    idx = get_intersection(list(idx1), list(idx2))
    idx = get_intersection(list(idx), list(idx3))
    return x[idx, :]

def get_longitudes(X, longitude, eps=0.5):
    eps = math.radians(eps)
    longitude = math.radians(longitude)
    R, theta, phi = cart2polar(X)
    idx = (phi >= longitude-eps) & (phi <= longitude+eps)
    idx = np.where(idx == True)[0]
    return idx

def get_latitudes(X, latitude, eps=0.5):
    eps = math.radians(eps)
    latitude = math.radians(latitude)
    R, theta, phi = cart2polar(X)
    idx = (theta >= latitude-eps) & (theta <= latitude+eps)
    idx = np.where(idx == True)[0]
    return idx

def place_landmarks(nlm, limits):
    landmarks = []
    x, y, z = polar2cart(R=1, theta=limits['theta_min'], phi=limits['phi_max']/2)
    landmarks.append(np.array([x, y, z]))
    x, y, z = polar2cart(R=1, theta=limits['theta_max']-5, phi=limits['phi_min'])
    landmarks.append(np.array([x, y, z]))
    x, y, z = polar2cart(R=1, theta=limits['theta_max']-5, phi=limits['phi_max']-5)
    landmarks.append(np.array([x, y, z]))

    if nlm >= 5:
        x, y, z = polar2cart(R=1, theta=limits['theta_max']/2, phi=limits['phi_min'])
        landmarks.append(np.array([x, y, z]))
        x, y, z = polar2cart(R=1, theta=limits['theta_max']/2, phi=limits['phi_max'])
        landmarks.append(np.array([x, y, z]))

    if nlm >=7:
        x, y, z = polar2cart(R=1, theta=limits['theta_max'], phi=limits['phi_max']/2)
        landmarks.append(np.array([x, y, z]))
        x, y, z = polar2cart(R=1, theta=limits['theta_max'] / 2, phi=limits['phi_max'] / 2)
        landmarks.append(np.array([x, y, z]))

    if nlm >= 9:
        x, y, z = polar2cart(R=1, theta= limits['theta_max'], phi=limits['phi_max']/4)
        landmarks.append(np.array([x, y, z]))
        x, y, z = polar2cart(R=1, theta= limits['theta_max'] , phi=3*limits['phi_max']/4)
        landmarks.append(np.array([x, y, z]))
    return landmarks

def generate_longitudes_latitudes(sphere, theta_grid, phi_grid):
    idx_lat = []
    idx_long = []
    for long in phi_grid:
        idx_long.append(get_longitudes(sphere, longitude=long))
    for lat in theta_grid:
        idx_lat.append(get_latitudes(sphere, latitude=lat))
    return idx_lat, idx_long