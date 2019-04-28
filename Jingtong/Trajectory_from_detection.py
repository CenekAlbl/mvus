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

shift_range = np.arange(0,6)
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
        estimate = ep.compute_fundamental_Ransac(x1,x2,threshold=2,maxiter=1500,loRansac=False)
        F = estimate['model'].reshape(3,3)
        F_svd = ep.compute_fundamental(x1[:,estimate['inliers']],x2[:,estimate['inliers']])
        F_cv, mask = ep.computeFundamentalMat(x1, x2, error=3)
        # F_all = ep.compute_fundamental(x1,x2)
        # model = ransac1.Ransac_Fundamental()
        # F_old, inliers = ransac1.F_from_Ransac(x1, x2, model, maxiter=500, threshold=2, inliers=100)

        # F_cv = F_cv/F_cv[2,2]

        # print('\nF using Ransac: \n{}\n'.format(F))
        # print('F using SVD on inliers: \n{}\n'.format(F_svd))
        # print('F using OpenCV: \n{}\n'.format(F_cv))
        # print('F using SVD on all points: \n{}\n'.format(F_all))
        # print('F using old Ransac: \n{}\n'.format(F_old))

        # error = np.mean(ep.Sampson_error(x1[:,estimate['inliers']], x2[:,estimate['inliers']], F))
        # error_svd = np.mean(ep.Sampson_error(x1[:,estimate['inliers']], x2[:,estimate['inliers']], F_svd))
        error_cv = np.mean(ep.Sampson_error(x1[:,mask.T[0]==1], x2[:,mask.T[0]==1], F_cv))
        # error_all = np.mean(ep.Sampson_error(x1, x2, F_all))
        # error_old = np.mean(ep.Sampson_error(x1[:,inliers], x2[:,inliers], F_old))
        # print('Error: {}, {}, {}, {}, {}'.format(error, error_svd, error_cv, error_all, error_old))

        print('Error: {}'.format(error_cv))
        # print('\nRatio of inliers: {:.3f}\n'.format(len(estimate['inliers']) / num_traj))
        print('Ratio of inliers from OpenCV: {:.3f}'.format(sum(mask.T[0]==1) / num_traj))
    else:
        F = ep.compute_fundamental(x1,x2)

    # Compute Beta and F
    # param = {'k':1, 's':0}
    # beta, F_beta, inliers = synchronization.iter_sync(x1,x2,param,p_max=7,threshold=5,maxiter=500,loRansac=False)
    # F_beta = F_beta.reshape((3,3))

    # Show epipolar lines
    idx = np.random.choice(num_traj,size=100,replace=False)
    vis.plot_epipolar_line(img1,img2,F,   x1[:,idx],x2[:,idx])
    vis.plot_epipolar_line(img1,img2,F_cv,x1[:,idx],x2[:,idx])

    # Compute epipole
    e_r = ep.compute_epipole_from_F(F)
    e_l = ep.compute_epipole_from_F(F,left=True)
    print('\nLeft epipole: {}\n'.format(e_l))
    print('Right epipole: {}\n'.format(e_r))

    e_r_cv = ep.compute_epipole_from_F(F_cv)
    e_l_cv = ep.compute_epipole_from_F(F_cv,left=True)
    print('\nLeft epipole from OpenCV: {}\n'.format(e_l_cv))
    print('Right epipole from OpenCV: {}\n'.format(e_r_cv))

    # Compute focal length
    # k1,k2 = ep.focal_length_from_F(F)
    # print('\nThe two estimated focal lengths are {:.3f} and {:.3f}\n'.format(k1,k2))

    k1,k2 = ep.focal_length_from_F(F_cv)
    print('The two estimated focal lengths are {:.3f} and {:.3f}'.format(k1,k2))
    
    # Triangulation 
    # E = np.dot(np.dot(K2.T,F),K1)
    # P1 = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    # X1,P2_1 = ep.triangulate_from_E(E,K1,K2,x1,x2)
    # X2,P2_2 = ep.triangulate_cv(E,K1,K2,x1,x2)

    # Triangulation OpenCV
    E = np.dot(np.dot(K2.T,F_cv),K1)
    P1 = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    X4,P2_4 = ep.triangulate_from_E(E,K1,K2,x1,x2)
    X5,P2_5 = ep.triangulate_cv(E,K1,K2,x1,x2)

    # Reprojection
    # h1_1 = np.dot(P1,X1)
    # h1_1 = h1_1/h1_1[-1]
    # h1_2 = np.dot(P2_1,X1)
    # h1_2 = h1_2/h1_2[-1]

    # h2_1 = np.dot(P1,X2)
    # h2_1 = h2_1/h2_1[-1]
    # h2_2 = np.dot(P2_2,X2)
    # h2_2 = h2_2/h2_2[-1]

    h4_1 = np.dot(P1,X4)
    h4_1 = h4_1/h4_1[-1]
    h4_2 = np.dot(np.dot(K2,P2_4),X4)
    h4_2 = h4_2/h4_2[-1]

    h5_1 = np.dot(P1,X5)
    h5_1 = h5_1/h5_1[-1]
    h5_2 = np.dot(P2_5,X5)
    h5_2 = h5_2/h5_2[-1]

    # Reprojection error
    # r1_1, r1_2 = np.mean(ep.reprojection_error(x1,h1_1)), np.mean(ep.reprojection_error(x2,h1_2))
    # r2_1, r2_2 = np.mean(ep.reprojection_error(x1,h2_1)), np.mean(ep.reprojection_error(x2,h2_2))
    r4_1, r4_2 = np.mean(ep.reprojection_error(x1,h4_1)), np.mean(ep.reprojection_error(x2,h4_2))
    r5_1, r5_2 = np.mean(ep.reprojection_error(x1,h5_1)), np.mean(ep.reprojection_error(x2,h5_2))

    # print('\nReprojection error 1: {}, {}'.format(r1_1,r1_2))
    # print('\nReprojection error 2: {}, {}'.format(r2_1,r2_2))
    print('\nReprojection error 4: {}, {}'.format(r4_1,r4_2))
    print('\nReprojection error 5: {}, {}'.format(r5_1,r5_2))

    # Show 3D results
    # vis.show_trajectory_3D(X1,X2,title='F from Jingtong, Triangulation left: from Jingtong, right: from OpenCV')

    # Show 3D results
    vis.show_trajectory_3D(X4,X5,title='F from OpenCV, Triangulation left: from Jingtong, right: from OpenCV',line=False)

    it += 1

print('Finished')