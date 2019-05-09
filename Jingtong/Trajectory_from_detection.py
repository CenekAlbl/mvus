import numpy as np
import cv2
import epipolar as ep
import visualization as vis
import util
import pickle
import ransac1
import synchronization
from scipy.interpolate import splprep, splev
from scipy.interpolate import UnivariateSpline
import scipy.io as scio
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
This script computes the 3D trajectory directly from 2D detection

Results using Ransac seems to be unstable: zigzag effects in different levels, different shape, etc.
'''

# Use Ransac or not
Ransac = True

# Compute focal length
focal_length = False

# Define shifting of trajectory (153,71,119)
start_1 = 153
start_2 = 118
num_traj = 1500

# Load calibration
calibration = open('data/calibration.pickle','rb')
K1 = np.asarray(pickle.load(calibration)["intrinsic_matrix"])
K2 = K1
camera1 = scio.loadmat('data/calibration/first_flight/gopro/calibration_narrow.mat')
K1 = camera1['intrinsic']
camera2 = scio.loadmat('data/calibration/first_flight/phone_2/calibration.mat')
K2 = camera2['intrinsic']

# Load data
traj_1 = np.loadtxt('data/video_1_output.txt',skiprows=1,dtype=np.int32)
traj_2 = np.loadtxt('data/video_3_output.txt',skiprows=1,dtype=np.int32)

x1 = np.vstack((traj_1[start_1:start_1+num_traj,1:].T, np.ones(num_traj)))
x2 = np.vstack((traj_2[start_2:start_2+num_traj,1:].T, np.ones(num_traj)))

# Create images for trajectory
r,c = 1080,1920
x1_int, x2_int = np.int16(x1), np.int16(x2)
img1 = np.zeros((r,c),dtype=np.uint8)
img2 = np.zeros((r,c),dtype=np.uint8)
for i in range(num_traj):
    img1[x1_int[1,i],x1_int[0,i]]= 255
    img2[x2_int[1,i],x2_int[0,i]]= 255

# Shift origin of image coordinate to the center
if focal_length:
    center_cam_1, center_cam_2 = np.array([1920/2,1080/2,0]), np.array([1920/2,1080/2,0])
    x1 = x1 - np.tile(center_cam_1,(num_traj,1)).T
    x2 = x2 - np.tile(center_cam_2,(num_traj,1)).T

# Visualize 2D trajectories
# vis.show_trajectory_2D(x1,x2,title='Without shift')    

# Spline fitting:
k, s = 3, 1000
x1_s = util.spline_fitting(x1[0], np.arange(0,num_traj,0.1), k=k, s=s)
y1_s = util.spline_fitting(x1[1], np.arange(0,num_traj,0.1), k=k, s=s)
x2_s = util.spline_fitting(x2[0], np.arange(0,num_traj,0.1), k=k, s=s)
y2_s = util.spline_fitting(x2[1], np.arange(0,num_traj,0.1), k=k, s=s)

# vis.show_spline((x1, np.vstack((x1_s,y1_s))), (x2, np.vstack((x2_s,y2_s))), title='k={}, s={}'.format(k,s))

shift_range = np.arange(-1,0)
it = 0
while it < len(shift_range):
    print('\n\n------------------Iteration {}------------------\n'.format(it+1))
    print('Current shift: {}'.format(shift_range[it]))
    
    # Define shift
    shift = shift_range[it]
    x1 = np.vstack((traj_1[start_1+shift:start_1+shift+num_traj,1:].T, np.ones(num_traj)))
    x2 = np.vstack((traj_2[start_2:start_2+num_traj,1:].T, np.ones(num_traj)))
 
    # Compute F
    if Ransac:
        # estimate = ep.compute_fundamental_Ransac(x1,x2,threshold=5,maxiter=1500,loRansac=False)
        # F = estimate['model'].reshape(3,3)
        # F_svd = ep.compute_fundamental(x1[:,estimate['inliers']],x2[:,estimate['inliers']])
        F_cv, mask = ep.computeFundamentalMat(x1, x2, error=5)

        # error = np.mean(ep.Sampson_error(x1[:,estimate['inliers']], x2[:,estimate['inliers']], F))
        error_cv = np.mean(ep.Sampson_error(x1[:,mask.T[0]==1], x2[:,mask.T[0]==1], F_cv))

        print('Error: {}'.format(error_cv))
        # print('\nRatio of inliers: {:.3f}\n'.format(len(estimate['inliers']) / num_traj))
        print('Ratio of inliers from OpenCV: {:.3f}'.format(sum(mask.T[0]==1) / num_traj))
    else:
        F = ep.compute_fundamental(x1,x2)

    # Compute Beta and F
    # param = {'k':1, 's':0}
    # beta, F_beta, inliers = synchronization.iter_sync(x1,x2,param,p_max=2,threshold=5,maxiter=500,loRansac=False)
    # F_cv = F_beta.reshape((3,3))

    # param = {'k':1, 's':0}
    # beta = -0.2
    # x2_shift = synchronization.shift_trajectory(x2,beta,k=param['k'],s=param['s'])
    # if beta >= 1:
    #     x2 = x2_shift[:,:-int(beta)]
    #     x1 = x1[:,:-int(beta)]
    # else:
    #     x2 = x2_shift[:,-int(beta):]
    #     x1 = x1[:,-int(beta):]


    # Show epipolar lines
    idx = np.random.choice(num_traj,size=100,replace=False)
    # vis.plot_epipolar_line(img1,img2,F,   x1[:,idx],x2[:,idx])
    vis.plot_epipolar_line(img1,img2,F_cv,x1[:,idx],x2[:,idx])

    # Compute epipole
    # e_r = ep.compute_epipole_from_F(F)
    # e_l = ep.compute_epipole_from_F(F,left=True)
    # print('\nLeft epipole: {}\n'.format(e_l))
    # print('Right epipole: {}\n'.format(e_r))

    e_r_cv = ep.compute_epipole_from_F(F_cv)
    e_l_cv = ep.compute_epipole_from_F(F_cv,left=True)
    print('\nLeft epipole from OpenCV: {}\n'.format(e_l_cv))
    print('Right epipole from OpenCV: {}\n'.format(e_r_cv))

    # Compute focal length
    p1 = K1[:,-1]
    p2 = K2[:,-1]
    f1_g, f2_g = K1[0,0],K2[0,0]

    k1 = np.sqrt(ep.focal_length_from_F_and_P(F_cv  ,p1,p2))
    k2 = np.sqrt(ep.focal_length_from_F_and_P(F_cv.T,p2,p1))
    k3,k4 = np.sqrt(ep.focal_length_iter(x1,x2,p1,p2,f1_g,f2_g))
    print('The two estimated focal lengths from Bougnoux  are {:.3f} and {:.3f}'.format(k1,k2))
    print('The two estimated focal lengths from iterative are {:.3f} and {:.3f}'.format(k3,k4))
    
    # Triangulation OpenCV
    E = np.dot(np.dot(K2.T,F_cv),K1)
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    X1,P2_1 = ep.triangulate_from_E(E,K1,K2,x1,x2)
    # X2,P2_2 = ep.triangulate_cv(E,K1,K2,x1,x2)
    X2,P2_2 = cv2.triangulatePoints(np.dot(K1,P1),np.dot(K2,P2_1),x1[:2],x2[:2]), P2_1
    X3      = ep.triangulate_matlab(np.dot(np.linalg.inv(K1),x1), np.dot(np.linalg.inv(K2),x2), P1, P2_2)

    # Reprojection
    h1_1 = np.dot(np.dot(K1,P1),X1)
    h1_1 = h1_1/h1_1[-1]
    h1_2 = np.dot(np.dot(K2,P2_1),X1)
    h1_2 = h1_2/h1_2[-1]

    h2_1 = np.dot(np.dot(K1,P1),X2)
    h2_1 = h2_1/h2_1[-1]
    h2_2 = np.dot(np.dot(K2,P2_2),X2)
    h2_2 = h2_2/h2_2[-1]

    h3_1 = np.dot(np.dot(K1,P1),X3)
    h3_1 = h3_1/h3_1[-1]
    h3_2 = np.dot(np.dot(K2,P2_2),X3)
    h3_2 = h3_2/h3_2[-1]

    # Reprojection error
    r1_1, r1_2 = np.mean(ep.reprojection_error(x1,h1_1)), np.mean(ep.reprojection_error(x2,h1_2))
    r2_1, r2_2 = np.mean(ep.reprojection_error(x1,h2_1)), np.mean(ep.reprojection_error(x2,h2_2))
    r3_1, r3_2 = np.mean(ep.reprojection_error(x1,h3_1)), np.mean(ep.reprojection_error(x2,h3_2))

    print('\nReprojection error 1: {}, {}'.format(r1_1,r1_2))
    print('\nReprojection error 2: {}, {}'.format(r2_1,r2_2))
    print('\nReprojection error 3: {}, {}'.format(r3_1,r3_2))

    # Show 3D results
    # vis.show_trajectory_3D(X1,X2,title='F from Jingtong, Triangulation left: from Jingtong, right: from OpenCV')

    # Show 3D results
    vis.show_trajectory_3D(X1,X2,X3,title='F from OpenCV, Triangulation left: from Jingtong, right: from OpenCV',line=False)

    it += 1

print('Finished')