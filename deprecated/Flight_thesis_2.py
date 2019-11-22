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
from scipy import interpolate, sparse



'''---------------New computation----------------'''
# Setting parameters
rows = 100000
error_F = 10
error_PnP = 30
cut_second = 0
cam_model = 6
sequence = [0,3,2]
smooth_factor = 0.001
sampling_rate = 0.02
outlier_thres = 50
tri_thres = 50
rs = True
setting = {'rows':rows, 'error_F':error_F, 'error_PnP':error_PnP, 'cut_second':cut_second, 'cam_model':cam_model, 'sequence':sequence,
           'smooth':smooth_factor, 'sampling':sampling_rate, 'outlier_thres':outlier_thres, 'tri_thres':tri_thres}

# Load FPS of each camera
fps1, fps2, fps3, fps4 = 29.970030, 29.838692, 25, 25

# Load camara intrinsic and radial distortions
intrin_1 = scio.loadmat('./data/paper/thesis/calibration/calib1.mat')
intrin_2 = scio.loadmat('./data/paper/thesis/calibration/calib2.mat')
intrin_3 = scio.loadmat('./data/paper/thesis/calibration/calib3.mat')
intrin_4 = scio.loadmat('./data/paper/thesis/calibration/calib4.mat')

K1, K2, K3, K4 = intrin_1['intrinsic'], intrin_2['intrinsic'], intrin_3['intrinsic'], intrin_4['intrinsic']
d1, d2, d3, d4 = intrin_1['radial_distortion'][0], intrin_2['radial_distortion'][0], intrin_3['radial_distortion'][0], intrin_4['radial_distortion'][0]
K4 = np.array([[1500,0,960],[0,1500,540],[0,0,1]],dtype='float')

cameras = [common.Camera(K=K1,d=d1,fps=fps1), common.Camera(K=K2,d=d2,fps=fps2), common.Camera(K=K3,d=d3,fps=fps3), \
           common.Camera(K=K4,d=d4,fps=fps4)]

# Load detections
detect_1 = np.loadtxt('./data/paper/thesis/detection/c1_f2.txt',usecols=(2,0,1))[:rows].T
detect_2 = np.loadtxt('./data/paper/thesis/detection/c2_f2.txt',usecols=(2,0,1))[:rows].T
detect_3 = np.loadtxt('./data/paper/thesis/detection/c3_f2.txt',usecols=(2,0,1))[:rows].T
detect_4 = np.loadtxt('./data/paper/thesis/detection/c4_f2.txt',usecols=(2,0,1))[:rows].T

# Create a scene
flight = common.Scene_multi_spline()
flight.setting = setting
flight.addCamera(*cameras)
flight.cam_model = cam_model
flight.addDetection(detect_1, detect_2, detect_3, detect_4)

# Initialize rolling shutter
flight.rs = np.zeros(flight.numCam, dtype=float) + rs * 1

# Add camera size
flight.cam_size = [np.array([1920,1080]), np.array([1920,1080]), np.array([1920,1080]), np.array([1920,1080])]

# Truncate detections
flight.cut_detection(second=cut_second)

# Define the order of cameras, the FIRST one will be the reference
flight.set_sequence(sequence)

# Add prior alpha and beta for each cameras
flight.init_alpha()
flight.beta = np.array([0.0, -714, -794.8, -384])

# Convert raw detections into the global timeline
flight.detection_to_global()

# Initialize the first 3D trajectory
flight.init_traj(error=error_F, inlier_only=False)

# Convert discrete trajectory to spline representation
flight.traj_to_spline(smooth_factor=smooth_factor)



'''---------------Incremental reconstruction----------------'''
start = datetime.now()
np.set_printoptions(precision=4)

cam_temp = 2
while True:
    print('\n----------------- Bundle Adjustment with {} cameras -----------------'.format(cam_temp))
    print('\nMean error of each camera before BA:   ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

    # Bundle adjustment
    res = flight.BA(cam_temp, rs=rs)

    print('\nMean error of each camera after BA:    ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

    flight.remove_outliers(flight.sequence[:cam_temp],thres=outlier_thres)

    res = flight.BA(cam_temp, rs=rs)

    print('\nMean error of each camera after second BA:    ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

    if cam_temp == len(sequence):
        print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
        break

    # Add the next camera and get its pose
    flight.get_camera_pose(flight.sequence[cam_temp],error=error_PnP)

    # Triangulate new points and update the 3D spline
    flight.triangulate(flight.sequence[cam_temp], flight.sequence[:cam_temp], thres=tri_thres, factor_t2s=smooth_factor, factor_s2t=sampling_rate)

    print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
    cam_temp += 1

# Visualize the 3D trajectory
flight.spline_to_traj(sampling_rate=1)
vis.show_trajectory_3D(flight.traj[1:],line=False)

# with open('./data/paper/thesis/trajectory/flight1.pkl','wb') as f:
#     pickle.dump(flight, f)

print('Finish !')