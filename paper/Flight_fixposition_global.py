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

fps1, fps2, fps3, fps4, fps5, fps6 = 29.727612, 50, 29.970030, 30.020690, 59.940060, 25

cameras = [common.Camera(K=K1,d=d1,fps=fps1), common.Camera(K=K2,d=d2,fps=fps2), common.Camera(K=K3,d=d3,fps=fps3), \
           common.Camera(K=K4,d=d4,fps=fps4), common.Camera(K=K5,d=d5,fps=fps5), common.Camera(K=K6,d=d6,fps=fps6)]

# Load detections
detect_1 = np.loadtxt('./data/paper/fixposition/detection/outp_mate10_1.txt',usecols=(2,0,1)).T
detect_2 = np.loadtxt('./data/paper/fixposition/detection/outp_sonyg1.txt',usecols=(2,0,1)).T
detect_3 = np.loadtxt('./data/paper/fixposition/detection/outp_sonyalpha5001.txt',usecols=(2,0,1)).T
detect_4 = np.loadtxt('./data/paper/fixposition/detection/outp_mate7_1.txt',usecols=(2,0,1)).T
detect_5 = np.loadtxt('./data/paper/fixposition/detection/outp_gopro1.txt',usecols=(2,0,1)).T
detect_6 = np.loadtxt('./data/paper/fixposition/detection/outp_sony5n1.txt',usecols=(2,0,1)).T

# Create a scene
flight = common.Scene_global_timeline()
flight.addCamera(*cameras)
flight.addDetection(detect_1, detect_2, detect_3, detect_4, detect_5, detect_6)

# Correct radial distortion, can be set to false
flight.undistort_detections()

# Define the order of cameras, the first one will be the reference
flight.set_sequence([0,1,2,3,4,5])

# Add prior alpha and beta for each cameras
flight.init_alpha()
flight.beta = np.array([0, 465.2250, -406.2928, -452.2184, 546.9844, 248.3173])

# Convert raw detections into the global timeline
flight.detection_to_global()

# Initialize the first 3D trajectory
flight.init_traj(error=30, inlier_only=False)

# Convert discrete trajectory to spline representation
flight.traj_to_spline(smooth_factor=0)

# Error of the first two cameras
error_0 = flight.error_cam(flight.sequence[0])
error_1 = flight.error_cam(flight.sequence[1])

# Backup befor BA
flight_b = copy.deepcopy(flight)

# Bundle adjustment
cam_temp = 2
res = flight.BA(cam_temp)

# Add the next camera and get its pose
flight.get_camera_pose(flight.sequence[cam_temp], verbose=1)

# Triangulate new points and update the 3D spline


