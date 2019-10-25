# This script trys to solve the BA bug for the 6th camera

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


with open('./data/paper/fixposition/flight_5cam_1.pkl', 'rb') as file:
    flight = pickle.load(file)

with open('./data/paper/fixposition/flight_6cam_1.pkl', 'rb') as file:
    flight_2 = pickle.load(file)

f1,f2,f3,f4,f5,f6 = flight.sequence[0],flight.sequence[1], flight.sequence[2], flight.sequence[3], flight.sequence[4], flight.sequence[5]

'''Add the sixth camera'''
flight.get_camera_pose(f6,error=30)
flight.error_cam(f6)

# Triangulate more points if possible
flight.triangulate_traj(f1,f6)
flight.triangulate_traj(f2,f6)
flight.triangulate_traj(f3,f6)
flight.triangulate_traj(f4,f6)
flight.triangulate_traj(f5,f6)

# Errors before optimization
flight.error_cam(0)
flight.error_cam(1)
flight.error_cam(2)
flight.error_cam(3)
flight.error_cam(4)
flight.error_cam(5)

# Define visibility
flight.set_visibility()

'''Optimize all 6 cameras'''
beta = flight.beta[0,(f1,f2,f3,f4,f5,f6)]
Track = flight.detections

cam_temp = [flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5],flight.cameras[f6]]
Track_temp = [Track[f1],Track[f2],Track[f3],Track[f4],Track[f5],Track[f6]]
traj_temp = copy.deepcopy(flight.traj)
v_temp = np.array([flight.visible[f1],flight.visible[f2],flight.visible[f3],flight.visible[f4],flight.visible[f5],flight.visible[f6]])
s_temp = copy.deepcopy(flight.spline)

res, model = common.optimize_all(cam_temp, Track_temp, traj_temp, v_temp, s_temp,
                                 include_K=True,max_iter=10,distortion=True,beta=beta)

flight.beta[0,(f1,f2,f3,f4,f5,f6)] = model[0]
flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5],flight.cameras[f6] = model[1][0], model[1][1], model[1][2], model[1][3], model[1][4], model[1][5]
flight.traj = model[2]

flight.undistort_detections(apply=True)
flight.set_tracks()

# Check reprojection error
print('\nAfter optimazing 6 cameras')
flight.error_cam(f1)
flight.error_cam(f2)
flight.error_cam(f3)
flight.error_cam(f4)
flight.error_cam(f5)
flight.error_cam(f6)


print('Finish!')