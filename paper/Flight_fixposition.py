import numpy as np
import util
import epipolar as ep
import synchronization
import common
import transformation
import scipy.io as scio
import pickle
import argparse
import copy
import cv2
import matplotlib.pyplot as plt
import visualization as vis
from datetime import datetime
from scipy.optimize import least_squares
from scipy import interpolate


'''---------------New computation----------------'''
# Load camara intrinsic and radial distortions
intrin_1 = scio.loadmat('./data/paper/fixposition/calibration/calib_mate10.mat')
intrin_2 = scio.loadmat('./data/paper/fixposition/calibration/calib_sonyg.mat')
intrin_3 = scio.loadmat('./data/paper/fixposition/calibration/calib_sony_alpha5100.mat')
intrin_4 = scio.loadmat('./data/paper/fixposition/calibration/calib_mate7.mat')
intrin_5 = scio.loadmat('./data/paper/fixposition/calibration/calib_gopro3.mat')
intrin_6 = scio.loadmat('./data/paper/fixposition/calibration/calib_sony_a5n.mat')

K1, K2, K3, K4, K5, K6 = intrin_1['intrinsic'], intrin_2['intrinsic'], intrin_3['intrinsic'], intrin_4['intrinsic'], intrin_5['intrinsic'], intrin_6['intrinsic']
d1, d2, d3, d4, d5, d6 = intrin_1['radial_distortion'][0], intrin_2['radial_distortion'][0], intrin_3['radial_distortion'][0], intrin_4['radial_distortion'][0], intrin_5['radial_distortion'][0], intrin_6['radial_distortion'][0]

# with open('./data/paper/fixposition/calibration/mate10/mate10_r.pickle', 'rb') as file:
#     a = pickle.load(file)
#     K1, d1 = a['intrinsic_matrix'], a['distortion_coefficients'][0,:2]
# with open('./data/paper/fixposition/calibration/sonyg/sonyg_r.pickle', 'rb') as file:
#     a = pickle.load(file)
#     K2, d2 = a['intrinsic_matrix'], a['distortion_coefficients'][0,:2]
# with open('./data/paper/fixposition/calibration/sony_alpha_5100/sony_alpha_5100_r.pickle', 'rb') as file:
#     a = pickle.load(file)
#     K3, d3 = a['intrinsic_matrix'], a['distortion_coefficients'][0,:2]
# with open('./data/paper/fixposition/calibration/mate7/mate7_r.pickle', 'rb') as file:
#     a = pickle.load(file)
#     K4, d4 = a['intrinsic_matrix'], a['distortion_coefficients'][0,:2]
# with open('./data/paper/fixposition/calibration/gopro3/gopro3_r.pickle', 'rb') as file:
#     a = pickle.load(file)
#     K5, d5 = a['intrinsic_matrix'], a['distortion_coefficients'][0,:2]
# with open('./data/paper/fixposition/calibration/sony_alpha_5n/sony_alpha_5n_r.pickle', 'rb') as file:
#     a = pickle.load(file)
#     K6, d6 = a['intrinsic_matrix'], a['distortion_coefficients'][0,:2]

cameras = [common.Camera(K=K1,d=d1), common.Camera(K=K2,d=d2), common.Camera(K=K3,d=d3), common.Camera(K=K4,d=d4), common.Camera(K=K5,d=d5), common.Camera(K=K6,d=d6)]

# Load detections
rows = 5000
detect_1 = np.loadtxt('./data/paper/fixposition/detection/outp_mate10_1_30.txt',usecols=(2,0,1))[:rows].T
detect_2 = np.loadtxt('./data/paper/fixposition/detection/outp_sonyg1_30.txt',usecols=(2,0,1))[:rows].T
detect_3 = np.loadtxt('./data/paper/fixposition/detection/outp_sonyalpha5001_30.txt',usecols=(2,0,1))[:rows].T
detect_4 = np.loadtxt('./data/paper/fixposition/detection/outp_mate7_1_30.txt',usecols=(2,0,1))[:rows].T
detect_5 = np.loadtxt('./data/paper/fixposition/detection/outp_gopro1_30.txt',usecols=(2,0,1))[:rows].T
detect_6 = np.loadtxt('./data/paper/fixposition/detection/outp_sony5n1_30.txt',usecols=(2,0,1))[:rows].T

# Create a scene
flight = common.Scene()
flight.addCamera(*cameras)
flight.addDetection(detect_1, detect_2, detect_3, detect_4, detect_5, detect_6)

# Correct radial distortion, can be set to false
flight.undistort_detections(apply=True)

# Compute beta for every pair of cameras
flight.beta = np.array([[0.0, -467.00, 409.49, 458.40, -552.17, -251.00]])
# flight.beta = -np.array([[0,470,-409,-461,553,252]])

# create tracks according to beta
flight.set_tracks()

# Sort detections in temporal order
flight.set_sequence()
flight.set_sequence([0,1,2,3,4,5])

# Set parameters manually
use_F = True
include_K = True
include_d = True
include_b = True
max_iter = 10
use_spline = False
smooth_factor = 0.01      # 0.005

if use_F:
    E_or_F = 'F'
    error_epip = 30
    error_PnP  = 30
else:
    E_or_F = 'E'
    error_epip = 0.1
    error_PnP  = 30

# Initialize the first 3D trajectory
idx1, idx2 = flight.init_traj(error=error_epip,F=use_F,inlier_only=False)

# Compute spline parameters and smooth the trajectory
if use_spline:
    flight.fit_spline(s=smooth_factor)
else:
    flight.spline = []

print('\nInitial cameras:', flight.sequence[:2])
print('Number of reconstructed points:', flight.traj.shape[1])
# vis.show_trajectory_3D(flight.traj[1:],line=False)


'''----------------Optimization----------------'''
start=datetime.now()

# Record settings
print('\nCurrently using F for the initial pair, K is optimized, beta and d are optimized, spline not applied')
print('Threshold for Epipolar:{}, Threshold for PnP:{}'.format(error_epip,error_PnP))

print('\nBefore optimization:')
f1,f2,f3,f4,f5,f6 = flight.sequence[0],flight.sequence[1], flight.sequence[2], flight.sequence[3], flight.sequence[4], flight.sequence[5]
flight.error_cam(f1)
flight.error_cam(f2)
flight_before = copy.deepcopy(flight)

'''Optimize two'''
res, model = common.optimize_two(flight.cameras[f1],flight.cameras[f2],flight.tracks[f1][1:,idx1],
                    flight.tracks[f2][1:,idx2],flight.traj,flight.spline,include_K=include_K,max_iter=max_iter)
flight.cameras[f1],flight.cameras[f2],flight.traj = model[0], model[1], model[2]

if use_spline:
    flight.spline = model[3]

# Check reprojection error
print('\nAfter optimizating first two cameras:')
flight.error_cam(f1)
flight.error_cam(f2)

print('\nTime: {}\n'.format(datetime.now()-start))

'''Add the third camera'''
flight.get_camera_pose(f3,error=error_PnP)
flight.error_cam(f3)

# Triangulate more points if possible
flight.triangulate_traj(f1,f3)
flight.triangulate_traj(f2,f3)

# Fit spline again if needed
if use_spline:
    flight.fit_spline(s=smooth_factor)

flight.error_cam(f1)
flight.error_cam(f2)
flight.error_cam(f3)

# Define visibility
flight.set_visibility()

'''Optimize all 3 cameras'''
# Before BA: set parameters
if include_b:
    beta = flight.beta[0,(f1,f2,f3)]
    if include_d:
        Track = flight.detections
    else:
        Track = flight.detections_undist
else:
    include_d = False
    beta = []
    Track = flight.tracks

# BA
cam_temp = [flight.cameras[f1],flight.cameras[f2],flight.cameras[f3]]
Track_temp = [Track[f1],Track[f2],Track[f3]]
traj_temp = copy.deepcopy(flight.traj)
v_temp = np.array([flight.visible[f1],flight.visible[f2],flight.visible[f3]])
s_temp = copy.deepcopy(flight.spline)

res, model = common.optimize_all(cam_temp, Track_temp, traj_temp, v_temp, s_temp,
                                 include_K=include_K,max_iter=max_iter,distortion=include_d,beta=beta)

# After BA: interpret results
if include_b:
    flight.beta[0,(f1,f2,f3)] = model[0]
    flight.cameras[f1],flight.cameras[f2],flight.cameras[f3] = model[1][0], model[1][1], model[1][2]
    flight.traj = model[2]
else:
    flight.cameras[f1],flight.cameras[f2],flight.cameras[f3] = model[0][0], model[0][1], model[0][2]
    flight.traj = model[1]

if use_spline:
    flight.spline = model[-1]

flight.undistort_detections(apply=True)
flight.set_tracks()

# Check reprojection error
print('\nAfter optimazing 3 cameras, beta:{}, d:{}'.format(include_b,include_d))
flight.error_cam(f1)
flight.error_cam(f2)
flight.error_cam(f3)

print('\nTime: {}\n'.format(datetime.now()-start))

'''Add the fourth camera'''
flight.get_camera_pose(f4,error=error_PnP)
flight.error_cam(f4)

# Triangulate more points if possible
flight.triangulate_traj(f1,f4)
flight.triangulate_traj(f2,f4)
flight.triangulate_traj(f3,f4)

# Fit spline again if needed
if use_spline:
    flight.fit_spline(s=smooth_factor)

flight.error_cam(f1)
flight.error_cam(f2)
flight.error_cam(f3)
flight.error_cam(f4)

# Define visibility
flight.set_visibility()

'''Optimize all 4 cameras'''
# Before BA: set parameters
if include_b:
    beta = flight.beta[0,(f1,f2,f3,f4)]
    if include_d:
        Track = flight.detections
    else:
        Track = flight.detections_undist
else:
    include_d = False
    beta = []
    Track = flight.tracks

# BA
cam_temp = [flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4]]
Track_temp = [Track[f1],Track[f2],Track[f3],Track[f4]]
traj_temp = copy.deepcopy(flight.traj)
v_temp = np.array([flight.visible[f1],flight.visible[f2],flight.visible[f3],flight.visible[f4]])
s_temp = copy.deepcopy(flight.spline)

res, model = common.optimize_all(cam_temp, Track_temp, traj_temp, v_temp, s_temp,
                                 include_K=include_K,max_iter=max_iter,distortion=include_d,beta=beta)

# After BA: interpret results
if include_b:
    flight.beta[0,(f1,f2,f3,f4)] = model[0]
    flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4] = model[1][0], model[1][1], model[1][2], model[1][3]
    flight.traj = model[2]
else:
    flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4] = model[0][0], model[0][1], model[0][2], model[0][3]
    flight.traj = model[1]

if use_spline:
    flight.spline = model[-1]

flight.undistort_detections(apply=True)
flight.set_tracks()

# Check reprojection error
print('\nAfter optimazing 4 cameras, beta:{}, d:{}'.format(include_b,include_d))
flight.error_cam(f1)
flight.error_cam(f2)
flight.error_cam(f3)
flight.error_cam(f4)

print('\nTime: {}\n'.format(datetime.now()-start))

'''Add the fifth camera'''
flight.get_camera_pose(f5,error=error_PnP)
flight.error_cam(f5)

# Triangulate more points if possible
flight.triangulate_traj(f1,f5)
flight.triangulate_traj(f2,f5)
flight.triangulate_traj(f3,f5)
flight.triangulate_traj(f4,f5)

# Fit spline again if needed
if use_spline:
    flight.fit_spline(s=smooth_factor)

flight.error_cam(f1)
flight.error_cam(f2)
flight.error_cam(f3)
flight.error_cam(f4)
flight.error_cam(f5)

# Define visibility
flight.set_visibility()

'''Optimize all 5 cameras'''
# Before BA: set parameters
if include_b:
    beta = flight.beta[0,(f1,f2,f3,f4,f5)]
    if include_d:
        Track = flight.detections
    else:
        Track = flight.detections_undist
else:
    include_d = False
    beta = []
    Track = flight.tracks

# BA
cam_temp = [flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5]]
Track_temp = [Track[f1],Track[f2],Track[f3],Track[f4],Track[f5]]
traj_temp = copy.deepcopy(flight.traj)
v_temp = np.array([flight.visible[f1],flight.visible[f2],flight.visible[f3],flight.visible[f4],flight.visible[f5]])
s_temp = copy.deepcopy(flight.spline)

res, model = common.optimize_all(cam_temp, Track_temp, traj_temp, v_temp, s_temp,
                                 include_K=include_K,max_iter=max_iter,distortion=include_d,beta=beta)

# After BA: interpret results
if include_b:
    flight.beta[0,(f1,f2,f3,f4,f5)] = model[0]
    flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5] = model[1][0], model[1][1], model[1][2], model[1][3], model[1][4]
    flight.traj = model[2]
else:
    flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f4] = model[0][0], model[0][1], model[0][2], model[0][3], model[0][4]
    flight.traj = model[1]

if use_spline:
    flight.spline = model[-1]

flight.undistort_detections(apply=True)
flight.set_tracks()

# Check reprojection error
print('\nAfter optimazing 5 cameras, beta:{}, d:{}'.format(include_b,include_d))
flight.error_cam(f1)
flight.error_cam(f2)
flight.error_cam(f3)
flight.error_cam(f4)
flight.error_cam(f5)

print('\nTime: {}\n'.format(datetime.now()-start))

'''Add the sixth camera'''
flight.get_camera_pose(f6,error=error_PnP)
flight.error_cam(f6)

# Triangulate more points if possible
flight.triangulate_traj(f1,f6)
flight.triangulate_traj(f2,f6)
flight.triangulate_traj(f3,f6)
flight.triangulate_traj(f4,f6)
flight.triangulate_traj(f5,f6)

# Fit spline again if needed
if use_spline:
    flight.fit_spline(s=smooth_factor)

flight.error_cam(f1)
flight.error_cam(f2)
flight.error_cam(f3)
flight.error_cam(f4)
flight.error_cam(f5)
flight.error_cam(f6)

# Define visibility
flight.set_visibility()

'''Optimize all 6 cameras'''
# Before BA: set parameters
if include_b:
    beta = flight.beta[0,(f1,f2,f3,f4,f5,f6)]
    if include_d:
        Track = flight.detections
    else:
        Track = flight.detections_undist
else:
    include_d = False
    beta = []
    Track = flight.tracks

# BA
cam_temp = [flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5],flight.cameras[f6]]
Track_temp = [Track[f1],Track[f2],Track[f3],Track[f4],Track[f5],Track[f6]]
traj_temp = copy.deepcopy(flight.traj)
v_temp = np.array([flight.visible[f1],flight.visible[f2],flight.visible[f3],flight.visible[f4],flight.visible[f5],flight.visible[f6]])
s_temp = copy.deepcopy(flight.spline)

res, model = common.optimize_all(cam_temp, Track_temp, traj_temp, v_temp, s_temp,
                                 include_K=include_K,max_iter=max_iter,distortion=include_d,beta=beta)

# After BA: interpret results
if include_b:
    flight.beta[0,(f1,f2,f3,f4,f5,f6)] = model[0]
    flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5],flight.cameras[f6] = model[1][0], model[1][1], model[1][2], model[1][3], model[1][4], model[1][5]
    flight.traj = model[2]
else:
    flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f4],flight.cameras[f6] = model[0][0], model[0][1], model[0][2], model[0][3], model[0][4], model[0][5]
    flight.traj = model[1]

if use_spline:
    flight.spline = model[-1]

flight.undistort_detections(apply=True)
flight.set_tracks()

# Check reprojection error
print('\nAfter optimazing 6 cameras, beta:{}, d:{}'.format(include_b,include_d))
flight.error_cam(f1)
flight.error_cam(f2)
flight.error_cam(f3)
flight.error_cam(f4)
flight.error_cam(f5)
flight.error_cam(f6)

print('\nTime: {}\n'.format(datetime.now()-start))

# with open('./data/paper/fixposition/flight_6cam.pkl','wb') as f:
#     pickle.dump(flight, f)

print('Finish !!')