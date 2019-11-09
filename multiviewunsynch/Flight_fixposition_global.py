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
# Setting parameters
rows = 100000
error_F = 30
cut_second = 0.5
sequence = [0,4,2,1,5,3]
smooth_factor = 0.0005
sampling_rate = 0.02

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
detect_1 = np.loadtxt('./data/paper/fixposition/detection/outp_mate10_1.txt',usecols=(2,0,1))[:rows].T
detect_2 = np.loadtxt('./data/paper/fixposition/detection/outp_sonyg1.txt',usecols=(2,0,1))[:rows].T
detect_3 = np.loadtxt('./data/paper/fixposition/detection/outp_sonyalpha5001.txt',usecols=(2,0,1))[:rows].T
detect_4 = np.loadtxt('./data/paper/fixposition/detection/outp_mate7_1.txt',usecols=(2,0,1))[:rows].T
detect_5 = np.loadtxt('./data/paper/fixposition/detection/outp_gopro1.txt',usecols=(2,0,1))[:rows].T
detect_6 = np.loadtxt('./data/paper/fixposition/detection/outp_sony5n1.txt',usecols=(2,0,1))[:rows].T

# Create a scene
flight = common.Scene_multi_spline()
flight.addCamera(*cameras)
flight.addDetection(detect_1, detect_2, detect_3, detect_4, detect_5, detect_6)

# Truncate detections
flight.cut_detection(second=cut_second)

# Define the order of cameras, the FIRST one will be the reference
flight.set_sequence(sequence)

# Add prior alpha and beta for each cameras
flight.init_alpha()
flight.beta = np.array([0, 465.2250, -406.2928, -452.2184, 546.9844, 248.3173])

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
    res = flight.BA(cam_temp)

    print('\nMean error of each camera after BA:    ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

    if cam_temp == len(sequence):
        print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
        break

    # Add the next camera and get its pose
    flight.get_camera_pose(flight.sequence[cam_temp])

    # Triangulate new points and update the 3D spline
    flight.triangulate(flight.sequence[cam_temp], flight.sequence[:cam_temp], factor_t2s=smooth_factor, factor_s2t=sampling_rate)

    print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
    cam_temp += 1

# Visualize the 3D trajectory
flight.spline_to_traj(sampling_rate=1)
vis.show_trajectory_3D(flight.traj[1:],line=False)

# with open('./data/paper/fixposition/trajectory/flight_spline_2.pkl','wb') as f:
#     pickle.dump(flight, f)

print('Finish !')