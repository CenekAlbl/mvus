import pickle
import cv2
import numpy as np
import epipolar as ep

'''
This script verifies the functionality of computing fundamental matrix F

in two ways (OpenCV and own function). As input a synthetic dataset is used

so that the resulting F should be nearly perfect.
'''

# Decide whether to add noises on point correspondences.
add_noise = True
noise_mean = 0
noise_std = 0

# Load trajectory data
X = np.loadtxt('data/Synthetic_Trajectory.txt')
X_homo = np.insert(X.T,3,1,axis=0)

# Load camera parameter
with open('data/Synthetic_Camera.pickle', 'rb') as file:
    Camera =pickle.load(file)

# Camara 1
P1 = np.dot(Camera["K1"],np.hstack((Camera["R1"],Camera["t1"])))
x1 = np.dot(P1,X_homo)
x1 /= x1[-1]
img1 = Camera["img1"]

# Camara 2
P2 = np.dot(Camera["K2"],np.hstack((Camera["R2"],Camera["t2"])))
x2 = np.dot(P2,X_homo)
x2 /= x2[-1]
img2 = Camera["img2"]

# Add noise (optional)
if add_noise:
    x1[:2] = x1[:2] + np.random.randn(2, x1.shape[1]) * noise_std
    x2[:2] = x2[:2] + np.random.randn(2, x2.shape[1]) * noise_std

    x1[0][x1[0]>img1.shape[1]] = img1.shape[1]
    x1[1][x1[1]>img1.shape[0]] = img1.shape[0]
    x1[0][x1[0]<0] = 0
    x1[1][x1[1]<0] = 0
    x2[0][x2[0]>img2.shape[1]] = img2.shape[1]
    x2[1][x2[1]>img2.shape[0]] = img2.shape[0]
    x2[0][x2[0]<0] = 0
    x2[1][x2[1]<0] = 0

# Use a certain percentage of data to compute F, the rest for validation
part = 0.7
num_all = x1.shape[1]
num_train = int(part*num_all)
num_val = num_all - num_train

idx = np.random.permutation(num_all)
x1_train = x1[:,idx[:num_train]]
x1_val = x1[:,idx[num_train:]]
x2_train = x2[:,idx[:num_train]]
x2_val = x2[:,idx[num_train:]]
print("\n{} points are used to compute F, {} points to validate using Sampson distance".format(num_train,num_val))

# Compute F in two ways
F1,mask = cv2.findFundamentalMat(x1_train[:2].T, x2_train[:2].T, method=cv2.FM_8POINT)
F2 = ep.compute_fundamental(x1_train,x2_train)

# Compute errors and compare
error_1 = ep.Sampson_error(x1_val, x2_val, F1)
error_2 = ep.Sampson_error(x1_val, x2_val, F2)

print("\nThe maxmial residual from OpenCV is {}".format(error_1.max()))
print("\nThe maxmial residual from Own implementation is {}".format(error_2.max()))
print("\nFinished\n")

# Visualize epipolar lines
ep.plot_epipolar_line(img1,img2,F2,x1_val,x2_val)