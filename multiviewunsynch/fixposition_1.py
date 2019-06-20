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
import visualization as vis
from datetime import datetime
from scipy.optimize import least_squares
from scipy import interpolate

'''----------------Previous results----------------'''
# Load ground truth 
gt = np.loadtxt('./data/fixposition/fixposition_1_xyz.txt').T

# Load previous computed flight
with open('./data/fixposition/flight_1.pkl', 'rb') as file:
    flight_pre = pickle.load(file)

# Load previous computed flight 30
with open('./data/fixposition/flight_1_30.pkl', 'rb') as file:
    flight_pre = pickle.load(file)


'''----Dense sample----'''
m = 10
traj_1 = flight_pre.traj[1:,642:]
traj_1_raw = traj_1
traj_2 = gt[:,3630:4040]

tck,u = interpolate.splprep(traj_1,u=np.arange(traj_1.shape[1]),s=0,k=1)
idx_1 = np.linspace(u[0],u[-1],traj_1.shape[1]*m)
x,y,z = interpolate.splev(idx_1,tck)
traj_1 = np.array([x,y,z])

idx_2 = np.arange(0,6*m*traj_2.shape[1],6*m)
error_min = 100

for i in range(traj_1.shape[1]-idx_2[-1]-1):
    idx_3 = idx_1[idx_2+i]
    x,y,z = interpolate.splev(idx_3,tck)
    traj_3 = np.array([x,y,z])

    # Estimate a similarity transformation to align trajectories
    M = transformation.affine_matrix_from_points(traj_3,traj_2,shear=False,scale=True)
    traj_4 = np.dot(M,util.homogeneous(traj_3))
    traj_4 /= traj_4[-1]

    # Evaluation
    scale, shear, angles, translate, perspective = transformation.decompose_matrix(M)
    error = np.sqrt((traj_2[0]-traj_4[0])**2 + (traj_2[1]-traj_4[1])**2 + (traj_2[2]-traj_4[2])**2)
    if np.mean(error) < error_min:
        error_min = np.mean(error)
        k = i
        M_best = M
        traj_4_best = traj_4

print('Mean error between transformed reconstruction and GPS data: {:.5f}, unit is meter.'.format(error_min))
# vis.show_trajectory_3D(traj_1_raw,traj_2,line=False,title='Raw Reconstruction vs GPS (1st flight)')
# vis.show_trajectory_3D(traj_4_best,traj_2,line=False,title='Reconstruction downsampled and transformed(similarity) vs GPS (1st flight)')


'''----Normal sample----'''
# Resample trajectories
# flight_1:     642,(3623,4048)
# flight_1_30:  649,(3622,4048)
traj_1 = flight_pre.traj[1:,649:]
traj_2 = gt[:,3622:4048]

tck,u = interpolate.splprep(traj_1,u=np.arange(traj_1.shape[1]),s=0,k=1)
x,y,z = interpolate.splev(np.linspace(u[0],u[-1],traj_2.shape[1]),tck)
traj_3 = np.array([x,y,z])

# Estimate a similarity transformation to align trajectories
M = transformation.affine_matrix_from_points(traj_3,traj_2,shear=False)
traj_4 = np.dot(M,util.homogeneous(traj_3))
traj_4 /= traj_4[-1]

# Evaluation
scale, shear, angles, translate, perspective = transformation.decompose_matrix(M)
error = np.sqrt((traj_2[0]-traj_4[0])**2 + (traj_2[1]-traj_4[1])**2 + (traj_2[2]-traj_4[2])**2)
error_x,error_y,error_z = np.sqrt((traj_2[0]-traj_4[0])**2), np.sqrt((traj_2[1]-traj_4[1])**2), np.sqrt((traj_2[2]-traj_4[2])**2)
print('Mean error between transformed reconstruction and GPS data: {:.5f}, unit is meter.'.format(np.mean(error)))

# Visualization
vis.show_trajectory_3D(traj_1,traj_2,line=False,title='Raw Reconstruction vs GPS (1st flight)')
vis.show_trajectory_3D(traj_4,traj_2,line=False,title='Reconstruction downsampled and transformed(similarity) vs GPS (1st flight)')


'''---------------New computation----------------'''
# Load camara intrinsic and radial distortions
intrin_1 = scio.loadmat('./data/calibration/fixposition/cam1/calibration.mat')
intrin_2 = scio.loadmat('./data/calibration/fixposition/cam2/calibration.mat')
intrin_3 = scio.loadmat('./data/calibration/fixposition/cam3/calibration.mat')
intrin_4 = scio.loadmat('./data/calibration/fixposition/cam4/calibration.mat')

K1, K2, K3, K4 = intrin_1['intrinsic'], intrin_2['intrinsic'], intrin_3['intrinsic'], intrin_4['intrinsic']
d1, d2, d3, d4 = intrin_1['radial_distortion'][0], intrin_2['radial_distortion'][0], intrin_3['radial_distortion'][0], intrin_4['radial_distortion'][0]
cameras = [common.Camera(K=K1,d=d1), common.Camera(K=K2,d=d2), common.Camera(K=K3,d=d3), common.Camera(K=K4,d=d4)]

# Load detections
detect_1 = np.loadtxt('./data/fixposition/detections/c1_f1_30.txt',usecols=(2,0,1)).T
detect_2 = np.loadtxt('./data/fixposition/detections/c2_f1_30.txt',usecols=(2,0,1)).T
detect_3 = np.loadtxt('./data/fixposition/detections/c3_f1.txt',usecols=(2,0,1)).T
detect_4 = np.loadtxt('./data/fixposition/detections/c4_f1.txt',usecols=(2,0,1)).T

# Create a scene
flight = common.Scene()
flight.addCamera(*cameras)
flight.addDetection(detect_1, detect_2, detect_3, detect_4)

# Correct radial distortion, can be set to false
flight.undistort_detections(apply=True)

# Compute beta for every pair of cameras
flight.beta = np.array([[0,-42,-574.4,-76.4],
                        [42,0,-532.4,-34.4],
                        [574.4,532.4,0,498],
                        [76.4,34.4,498,0]])
# flight.compute_beta(threshold_error=2)

# create tracks according to beta
flight.set_tracks()

# Sort detections in temporal order
flight.set_sequence()
flight.set_sequence([0,1,3,2])

# Set parameters manually
use_F = True
include_K = True
include_d = True
include_b = True
max_iter = 15
use_spline = False
smooth_factor = 0.005

if use_F:
    E_or_F = 'F'
    error_epip = 60
    error_PnP  = 80
else:
    E_or_F = 'E'
    error_epip = 0.1
    error_PnP  = 30

# Initialize the first 3D trajectory
idx1, idx2 = flight.init_traj(error=error_epip,F=use_F,inlier_only=True)

# Compute spline parameters and smooth the trajectory
if use_spline:
    flight.fit_spline(s=smooth_factor)
else:
    flight.spline = []


'''----------------Optimization----------------'''
start=datetime.now()

# Record settings
print('\nCurrently using E for the initial pair, K is optimized, beta and d are optimized, spline not applied')
print('Threshold for Epipolar:{}, Threshold for PnP:{}'.format(error_epip,error_PnP))

print('\nBefore optimization:')
f1,f2,f3,f4 = flight.sequence[0],flight.sequence[1], flight.sequence[2], flight.sequence[3]
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
    beta = flight.beta[0]
    if include_d:
        Track = flight.detections
    else:
        Track = flight.detections_undist
else:
    include_d = False
    beta = []
    Track = flight.tracks

# BA
res, model = common.optimize_all(flight.cameras,Track,flight.traj,flight.visible,flight.spline,include_K=include_K,
                        max_iter=max_iter,distortion=include_d,beta=beta)

# After BA: interpret results
if include_b:
    flight.beta[0], flight.cameras, flight.traj = model[0], model[1], model[2]
else:
    flight.cameras, flight.traj = model[0], model[1]

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

# with open('./data/fixposition/flight_1.pkl','wb') as f:
#     pickle.dump(flight, f)

print('Finished')