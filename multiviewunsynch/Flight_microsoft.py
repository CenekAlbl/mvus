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
import cProfile


'''---------------New computation----------------'''
# Setting parameters
rows = 100000
error_F = 30
cut_second = 0.5
cam_model = 12
sequence = [0,1,3,5,2,4]
smooth_factor = 0.001
sampling_rate = 0.05
tri_thres = 20
setting = {'rows':rows, 'error_F':error_F, 'cut_second':cut_second, 'cam_model':cam_model, 'sequence':sequence, 'smooth':smooth_factor, 'sampling':sampling_rate, 'tri_thres':tri_thres}

# Load camara intrinsic and radial distortions
K1 = np.loadtxt('./data/paper/MS/calibration/cam_0.txt')
K2 = np.loadtxt('./data/paper/MS/calibration/cam_4.txt')
K3 = np.loadtxt('./data/paper/MS/calibration/cam_5.txt')
K4 = np.loadtxt('./data/paper/MS/calibration/cam_6.txt')
K5 = np.loadtxt('./data/paper/MS/calibration/cam_7.txt')
K6 = np.loadtxt('./data/paper/MS/calibration/cam_8.txt')

d = np.array([0,0],dtype=float)
fps = 30

cameras = [common.Camera(K=K1,d=d,fps=fps), common.Camera(K=K2,d=d,fps=fps), common.Camera(K=K3,d=d,fps=fps),
           common.Camera(K=K4,d=d,fps=fps), common.Camera(K=K5,d=d,fps=fps), common.Camera(K=K6,d=d,fps=fps)]

# Load detections
detect_1 = np.loadtxt('./data/paper/MS/detection/out_cam0.txt',usecols=(2,0,1))[:rows].T
detect_2 = np.loadtxt('./data/paper/MS/detection/out_cam04.txt',usecols=(2,0,1))[:rows].T
detect_3 = np.loadtxt('./data/paper/MS/detection/out_cam05.txt',usecols=(2,0,1))[:rows].T
detect_4 = np.loadtxt('./data/paper/MS/detection/out_cam06.txt',usecols=(2,0,1))[:rows].T
detect_5 = np.loadtxt('./data/paper/MS/detection/out_cam07.txt',usecols=(2,0,1))[:rows].T
detect_6 = np.loadtxt('./data/paper/MS/detection/out_cam08.txt',usecols=(2,0,1))[:rows].T

# Create a scene
flight = common.Scene_multi_spline()
flight.setting = setting
flight.addCamera(*cameras)
flight.cam_model = cam_model
flight.addDetection(detect_1, detect_2, detect_3, detect_4, detect_5, detect_6)

# Truncate detections
flight.cut_detection(second=cut_second)

# Define the order of cameras, the FIRST one will be the reference
flight.set_sequence(sequence)

# Add prior alpha and beta for each cameras
flight.init_alpha()
flight.beta = np.array([0,0,0,0,0,0], dtype=float)

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
    flight.triangulate(flight.sequence[cam_temp], flight.sequence[:cam_temp], thres=tri_thres, factor_t2s=smooth_factor, factor_s2t=sampling_rate)

    print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
    cam_temp += 1

# Visualize the 3D trajectory
flight.spline_to_traj(sampling_rate=1)
vis.show_trajectory_3D(flight.traj[1:],line=False)

# with open('./data/paper/fixposition/trajectory/flight_spline_2.pkl','wb') as f:
#     pickle.dump(flight, f)

print('Finish !')