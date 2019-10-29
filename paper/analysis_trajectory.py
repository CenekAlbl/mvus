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


with open('./data/paper/fixposition/trajectory/flight_5cam_1.pkl', 'rb') as file:
    flight = pickle.load(file)

with open('./data/paper/fixposition/trajectory/flight_6cam_1.pkl', 'rb') as file:
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
error_total_before = np.concatenate((flight.error_cam(f1,dist=False),flight.error_cam(f2,dist=False),flight.error_cam(f3,dist=False), \
                                     flight.error_cam(f4,dist=False),flight.error_cam(f5,dist=False),flight.error_cam(f6,dist=False)))
error_2 = np.concatenate((flight_2.error_cam(f1,dist=False),flight_2.error_cam(f2,dist=False),flight_2.error_cam(f3,dist=False), \
                          flight_2.error_cam(f4,dist=False),flight_2.error_cam(f5,dist=False),flight_2.error_cam(f6,dist=False)))

# Define visibility
flight.set_visibility()

# Visualize detections and reprojections
# for id_cam in flight.sequence:
#     inter,idx1,idx2 = np.intersect1d(flight.traj[0],flight.tracks[id_cam][0],assume_unique=True,return_indices=True)
#     X = util.homogeneous(flight.traj[1:,idx1])
#     x = util.homogeneous(flight.tracks[id_cam][1:,idx2])
#     x_cal = flight.cameras[id_cam].projectPoint(X)
#     vis.show_2D_all(x,x_cal,line=False)



'''Optimize all 6 cameras'''
beta = flight.beta[0,(f1,f2,f3,f4,f5,f6)]
Track = flight.detections

cam_temp = [flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5],flight.cameras[f6]]
Track_temp = [Track[f1],Track[f2],Track[f3],Track[f4],Track[f5],Track[f6]]
traj_temp = copy.deepcopy(flight.traj)
v_temp = np.array([flight.visible[f1],flight.visible[f2],flight.visible[f3],flight.visible[f4],flight.visible[f5],flight.visible[f6]])
s_temp = copy.deepcopy(flight.spline)

# Reproduce the error computation
mode = flight.beta.ravel()
for i in flight.cameras:
    mode = np.concatenate((mode,i.P2vector(n=11)))
mode = np.concatenate((mode,np.ravel(flight.traj[1:].T)))

num_Cam = 6
n_cam = 11
i=5

flight.cameras[i].info()
pp = flight.cameras[i].P2vector(n=6)
flight.cameras[i].vector2P(pp,n=6)
flight.cameras[i].info()

for i in range(num_Cam):
    flight.cameras[i].vector2P(mode[num_Cam+i*n_cam:num_Cam+(i+1)*n_cam],n=n_cam)

Tracks = common.detect_undistort(flight.detections,flight.cameras)
Tracks = common.detect_to_track(Tracks,mode[:num_Cam])
flight.traj[1:] = mode[num_Cam+num_Cam*n_cam:].reshape(-1,3).T

error = np.array([])
num_Point = flight.traj.shape[1]

error = np.array([])
for i in range(num_Cam):
    inter,idx1,idx2 = np.intersect1d(flight.traj[0],Tracks[i][0],assume_unique=True,return_indices=True)
    X_temp = util.homogeneous(flight.traj[1:,idx1])
    x = Tracks[i][1:,idx2]
    x_repro = flight.cameras[i].projectPoint(X_temp)

    error_cam_i = np.zeros(num_Point*2)
    error_cam_i[idx1],error_cam_i[idx1+num_Point] = abs(x_repro[0]-x[0]),abs(x_repro[1]-x[1])

    error = np.concatenate((error,error_cam_i))
hh = error

# BA
res, model = common.optimize_all(cam_temp, Track_temp, traj_temp, v_temp, s_temp,
                                 include_K=False,max_iter=10,distortion=True,beta=beta)

flight.beta[0,(f1,f2,f3,f4,f5,f6)] = model[0]
flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5],flight.cameras[f6] = model[1][0], model[1][1], model[1][2], model[1][3], model[1][4], model[1][5]
flight.traj = model[2]

flight.undistort_detections(apply=True)
flight.set_tracks()

# Check reprojection error
print('\nAfter optimazing 6 cameras')
error_total_after = np.concatenate((flight.error_cam(f1,dist=False),flight.error_cam(f2,dist=False),flight.error_cam(f3,dist=False), \
                                     flight.error_cam(f4,dist=False),flight.error_cam(f5,dist=False),flight.error_cam(f6,dist=False)))


print('Finish!')