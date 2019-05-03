import numpy as np
import cv2
import epipolar as ep
import visualization as vis
import util
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
This script tests the computation of focal length on a synthetic dataset
'''

# Settings
noise = True

# Load trajectory data
X = np.loadtxt('data/Synthetic_Trajectory_generated.txt')
X = np.insert(X,3,1,axis=0)
num_points = X.shape[1]

# Load camera parameter
with open('data/Synthetic_Camera.pickle', 'rb') as file:
    Camera =pickle.load(file)

# Camara 1
P1 = np.dot(Camera["K1"],np.hstack((Camera["R1"],Camera["t1"])))
x1 = np.dot(P1,X)
x1 /= x1[-1]
img1 = Camera["img1"]

# Camara 2
P2 = np.dot(Camera["K2"],np.hstack((Camera["R2"],Camera["t2"])))
x2 = np.dot(P2,X)
x2 /= x2[-1]
img2 = Camera["img2"]

# add noise
if noise:
    noise_std = 1
    noise_1 = np.random.randn(2, num_points) * noise_std
    noise_2 = np.random.randn(2, num_points) * noise_std
    x1[:2] += noise_1
    x2[:2] += noise_2

# Compute F
F, mask = ep.computeFundamentalMat(x1,x2,error=3)

# Compute focal length
p1 = Camera['K1'][:,-1]
p2 = Camera['K2'][:,-1]

# Bougnoux method
k1 = ep.focal_length_from_F_and_P(F,p1,p2)
k2 = ep.focal_length_from_F_and_P(F.T,p2,p1)

# Iterative method
f1_g, f2_g = 600,700
k3,k4 = ep.focal_length_iter(x1,x2,p1,p2,f1_g,f2_g)

# Show results
print('\nRatio of inliers for estimation of F: {}'.format(sum(mask.T[0]==1) / num_points))
print('\nTwo True focal lengths are {:.3f} and {:.3f}'.format(Camera["K1"][0,0], Camera["K2"][0,0]))
print('\nTwo estimated focal lengths from Bougnoux are {:.3f} and {:.3f}'.format(np.sqrt(k1), np.sqrt(k2)))
print('\nInitial guesses are {:.3f} and {:.3f}'.format(f1_g, f2_g))
print('\nTwo estimated focal lengths from Hartley  are {:.3f} and {:.3f}'.format(np.sqrt(k3), np.sqrt(k4)))

print('\nFinished\n')