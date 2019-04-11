import numpy as np
import cv2
import util
import pickle
import epipolar as ep
import synchronization
import visualization as vis

'''
This script tests the estimation of fundamental matrix F of the real trajectory data

with and without considering synchronization

!! Not Meaningful right now !!
'''

# Load data
start_traj_1 = 153
start_traj_2 = 71
num_traj = 1500

traj_1 = np.loadtxt('data/video_1_output.txt',skiprows=1,dtype=np.int32)
traj_2 = np.loadtxt('data/video_2_output.txt',skiprows=1,dtype=np.int32)

x1 = np.vstack((traj_1[start_traj_1:start_traj_1+num_traj,1:].T, np.ones(num_traj)))
x2 = np.vstack((traj_2[start_traj_2:start_traj_2+num_traj,1:].T, np.ones(num_traj)))

# Without synchronization
estimate1 = ep.compute_fundamental_Ransac(x1,x2,threshold=1,maxiter=500,loRansac=True)
F1 = estimate1['model'].reshape(3,3)
inliers1 = estimate1['inliers']


# With synchronization
estimate2 = synchronization.compute_beta_fundamental_Ransac(x1,x2,threshold=1,maxiter=500,loRansac=True)
F2 = estimate2['model'][:9].reshape(3,3)
beta = estimate2['model'][9]
inliers2 = estimate2['inliers']

# Load calibration matrix
calibration = open('data/calibration.pickle','rb')
K1 = np.asarray(pickle.load(calibration)["intrinsic_matrix"])
K2 = K1

'''Without Beta'''

# Compute essential matrix E from fundamental matrix F
E1 = np.dot(np.dot(K2.T,F1),K1)
num, R1, t1, mask = cv2.recoverPose(E1, x1[:2].T, x2[:2].T, K1)
P1_1 = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
P2_1 = np.dot(K2,np.hstack((R1,t1)))

print("\nTriangulating feature points...")
pts1_1 = x1[:2].astype(np.float64)
pts2_1 = x2[:2].astype(np.float64)
X1 = cv2.triangulatePoints(P1_1,P2_1,pts1_1,pts2_1)
X1/=X1[-1]

# Rescale
X1[0] = util.mapminmax(X1[0],-5,5)
X1[1] = util.mapminmax(X1[1],-5,5)
X1[2] = util.mapminmax(X1[2],-5,5)

# Show 3D results
vis.show_trajectory_3D(X1)

'''With Beta'''

# Compute essential matrix E from fundamental matrix F
E2 = np.dot(np.dot(K2.T,F2),K1)
num, R2, t2, mask = cv2.recoverPose(E2, x1[:2].T, x2[:2].T, K1)
P1_2 = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
P2_2 = np.dot(K2,np.hstack((R2,t2)))

print("\nTriangulating feature points...")
pts1_2 = x1[:2].astype(np.float64)
if abs(beta) < x2.shape[1]:
    x2_shift = synchronization.shift_trajectory(x2,beta)
    if beta >= 0:
        x2[:,:x2_shift.shape[1]] = x2_shift
    else:
        x2[:,x2.shape[1]-x2_shift.shape[1]:] = x2_shift
pts2_2 = x2[:2].astype(np.float64)

X2 = cv2.triangulatePoints(P1_2,P2_2,pts1_2,pts2_2)
X2/=X2[-1]

# Rescale
X2[0] = util.mapminmax(X2[0],-5,5)
X2[1] = util.mapminmax(X2[1],-5,5)
X2[2] = util.mapminmax(X2[2],-5,5)

# Show 3D results
vis.show_trajectory_3D(X2)

print('\nfinished\n')
    