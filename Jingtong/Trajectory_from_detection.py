import numpy as np
import cv2
import epipolar as ep
import visualization as vis
import util
import pickle
from scipy.interpolate import splprep, splev
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
This script computes the 3D trajectory directly from 2D detection

Results using Ransac seems to be unstable: zigzag effects in different levels, different shape, etc.
'''

# Use Ransac or not
Ransac = True

# Define shifting of trajectory
start_1 = 153
start_2 = 71
num_traj = 1500

# Load calibration
calibration = open('data/calibration.pickle','rb')
K1 = np.asarray(pickle.load(calibration)["intrinsic_matrix"])
K2 = K1

# Load data
traj_1 = np.loadtxt('data/video_1_output.txt',skiprows=1,dtype=np.int32)
traj_2 = np.loadtxt('data/video_2_output.txt',skiprows=1,dtype=np.int32)

x1 = np.vstack((traj_1[start_1:start_1+num_traj,1:].T, np.ones(num_traj)))
x2 = np.vstack((traj_2[start_2:start_2+num_traj,1:].T, np.ones(num_traj)))

# Spline fitting
# x_uniq = np.unique(x1[:2],axis=1)
# tck, u = splprep([x_uniq[0],x_uniq[1]], s=0)
# new = splev(u,tck)

# Visualize 2D trajectories
vis.show_trajectory_2D(x1,x2,title='Without shift')

# Compute F
if Ransac:
    estimate = ep.compute_fundamental_Ransac(x1,x2,threshold=10,maxiter=300,loRansac=True)
    F = estimate['model'].reshape(3,3)
else:
    F = ep.compute_fundamental(x1,x2)

# Compute focal length
k1,k2 = ep.focal_length_from_F(F)
print('\nThe two estimated focal lengths are {} and {}'.format(k1,k2))

# Triangulation
E = np.dot(np.dot(K2.T,F),K1)
num, R, t, mask = cv2.recoverPose(E, x1[:2].T, x2[:2].T, K1)
P1 = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
P2 = np.dot(K2,np.hstack((R,t)))

print("\nTriangulating feature points...")
pts1 = x1[:2].astype(np.float64)
pts2 = x2[:2].astype(np.float64)
X = cv2.triangulatePoints(P1,P2,pts1,pts2)
X/=X[-1]

# Rescale
X[0] = util.mapminmax(X[0],-5,5)
X[1] = util.mapminmax(X[1],-5,5)
X[2] = util.mapminmax(X[2],-5,5)

# Show 3D results
vis.show_trajectory_3D(X)

print('Finished')