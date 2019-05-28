import numpy as np
import cv2
import epipolar as ep
import visualization as vis
import util
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
This script tests different versions of triangulation on a synthetic dataset
'''

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

x1[:2] = x1[:2] + np.random.randn(2,num_points)*0.1
x2[:2] = x2[:2] + np.random.randn(2,num_points)*0.1

# Triangulation
kk = np.array([[0,0,0],[0,0,0],[0,0,0]])
K1,K2 = Camera["K1"]+kk, Camera["K2"]
F,inlier = ep.computeFundamentalMat(x1,x2,error=1)
E = np.dot(np.dot(K2.T,F),K1)
Y_E, P = ep.triangulate_from_E(E,K1,K2,x1,x2)
Y = ep.triangulate_matlab(x1,x2,P1,P2)
Y_n = ep.triangulate_matlab(np.dot(np.linalg.inv(Camera["K1"]),x1),np.dot(np.linalg.inv(Camera["K2"]),x2),np.hstack((Camera["R1"],Camera["t1"])),np.hstack((Camera["R2"],Camera["t2"])))

# Visualization
vis.show_trajectory_3D(Y,Y_E, line=False)

# Reprojection
y1_E, y2_E = np.dot(np.dot(Camera["K1"],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])),Y_E), np.dot(np.dot(Camera["K2"],P),Y_E)
y1_E, y2_E = y1_E/y1_E[-1], y2_E/y2_E[-1]
y1, y2 = np.dot(P1,Y), np.dot(P2,Y)
y1, y2 = y1/y1[-1], y2/y2[-1]
y1_n, y2_n = np.dot(P1,Y_n), np.dot(P2,Y_n)
y1_n, y2_n = y1_n/y1_n[-1], y2_n/y2_n[-1]

# Visualization
vis.show_trajectory_2D(y1_E,y2_E,y1,y2)

# Reprojection error
err_E1, err_E2 = np.mean(ep.reprojection_error(x1,y1_E)), np.mean(ep.reprojection_error(x2,y2_E))
err_1, err_2 = np.mean(ep.reprojection_error(x1,y1)), np.mean(ep.reprojection_error(x2,y2))
err_n_1, err_n_2 = np.mean(ep.reprojection_error(x1,y1_n)), np.mean(ep.reprojection_error(x2,y2_n))

print('Finish')