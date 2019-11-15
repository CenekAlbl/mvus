import numpy as np
import util
import epipolar as ep
import synchronization
import common_rs_mp as common
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
cam_model = 6
sequence = [0,4,2,1,5,3]       #[0,2,3,4,1,5]
smooth_factor = 0.001
sampling_rate = 0.02
rs_crt = False
sparse = True
outlier_thres = 6
tri_thres = 15
setting = {'rows':rows, 'error_F':error_F, 'cut_second':cut_second, 'cam_model':cam_model, 'sequence':sequence,
           'smooth':smooth_factor, 'sampling':sampling_rate, 'tri_thres':tri_thres, 'outlier_thres':outlier_thres}

# Load FPS of each camera
fps1, fps2, fps3, fps4, fps5, fps6 = 29.727612, 50, 29.970030, 30.020690, 59.940060, 25

# # Load camara intrinsic and radial distortions
# intrin_1 = scio.loadmat('./data/paper/fixposition/calibration/calib_mate10.mat')
# intrin_2 = scio.loadmat('./data/paper/fixposition/calibration/calib_sonyg.mat')
# intrin_3 = scio.loadmat('./data/paper/fixposition/calibration/calib_sony_alpha5100.mat')
# intrin_4 = scio.loadmat('./data/paper/fixposition/calibration/calib_mate7.mat')
# intrin_5 = scio.loadmat('./data/paper/fixposition/calibration/calib_gopro3.mat')
# intrin_6 = scio.loadmat('./data/paper/fixposition/calibration/calib_sony_a5n.mat')

# K1, K2, K3, K4, K5, K6 = intrin_1['intrinsic'], intrin_2['intrinsic'], intrin_3['intrinsic'], intrin_4['intrinsic'], intrin_5['intrinsic'], intrin_6['intrinsic']
# d1, d2, d3, d4, d5, d6 = intrin_1['radial_distortion'][0], intrin_2['radial_distortion'][0], intrin_3['radial_distortion'][0], intrin_4['radial_distortion'][0], intrin_5['radial_distortion'][0], intrin_6['radial_distortion'][0]
# d1, d2, d3, d4, d5, d6 = np.append(d1,0), np.append(d2,0), np.append(d3,0), np.append(d4,0), np.append(d5,0), np.append(d6,0)

# cameras = [common.Camera(K=K1,d=d1,fps=fps1), common.Camera(K=K2,d=d2,fps=fps2), common.Camera(K=K3,d=d3,fps=fps3), \
#            common.Camera(K=K4,d=d4,fps=fps4), common.Camera(K=K5,d=d5,fps=fps5), common.Camera(K=K6,d=d6,fps=fps6)]
# gopro = np.load('./data/paper/fixposition/calibration/gopro.npz')
# cameras[4].d = gopro.f.dist[0,[0,1,4]]
# cameras[4].tan = gopro.f.dist[0,[2,3]]

# Calibration from Opencv
c1 = np.load('data2/calib_fixpoition_6cam/mate10.npz')
c2 = np.load('data2/calib_fixpoition_6cam/sonyG.npz')
c3 = np.load('data2/calib_fixpoition_6cam/sony_alpha_5100.npz')
c4 = np.load('data2/calib_fixpoition_6cam/mate7.npz')
c5 = np.load('data2/calib_fixpoition_6cam/gopro.npz')
c6 = np.load('data2/calib_fixpoition_6cam/sony_alpha_5n.npz')

# Rolling Shutter correction 
rs_init = 0
# Number of Sensor lines for each camera (used for RS correction)
n1=  c1.f.mtx[1,2]*2
n2 = c2.f.mtx[1,2]*2
n3 = c3.f.mtx[1,2]*2
n4 = c4.f.mtx[1,2]*2
n5 = c5.f.mtx[1,2]*2
n6 = c6.f.mtx[1,2]*2

cameras = [common.Camera(K=c1.f.mtx,d=c1.f.dist[0,[0,1,4]],fps=fps1,rs=rs_init,ydim=n1,tan=c1.f.dist[0,[2,3]]),
           common.Camera(K=c2.f.mtx,d=c2.f.dist[0,[0,1,4]],fps=fps2,rs=rs_init,ydim=n2,tan=c2.f.dist[0,[2,3]]),
           common.Camera(K=c3.f.mtx,d=c3.f.dist[0,[0,1,4]],fps=fps3,rs=rs_init,ydim=n3,tan=c3.f.dist[0,[2,3]]),
           common.Camera(K=c4.f.mtx,d=c4.f.dist[0,[0,1,4]],fps=fps4,rs=rs_init,ydim=n4,tan=c4.f.dist[0,[2,3]]),
           common.Camera(K=c5.f.mtx,d=c5.f.dist[0,[0,1,4]],fps=fps5,rs=rs_init,ydim=n5,tan=c5.f.dist[0,[2,3]]),
           common.Camera(K=c6.f.mtx,d=c6.f.dist[0,[0,1,4]],fps=fps6,rs=rs_init,ydim=n6,tan=c6.f.dist[0,[2,3]])]

# Load detections
# Load detections
detect_1 = np.loadtxt('data2/detections/outp_mate10_1.txt',usecols=(2,0,1))[:rows].T
detect_2 = np.loadtxt('data2/detections/outp_sonyg1.txt',usecols=(2,0,1))[:rows].T
detect_3 = np.loadtxt('data2/detections/outp_sonyalpha5001.txt',usecols=(2,0,1))[:rows].T
detect_4 = np.loadtxt('data2/detections/outp_mate7_cleaned.txt',usecols=(2,0,1))[:rows].T
detect_5 = np.loadtxt('data2/detections/outp_gopro1.txt',usecols=(2,0,1))[:rows].T
detect_6 = np.loadtxt('data2/detections/outp_sony5n1.txt',usecols=(2,0,1))[:rows].T

# detect_1 = np.loadtxt('./data/paper/fixposition/detection/outp_mate10_1.txt',usecols=(2,0,1))[:rows].T
# detect_2 = np.loadtxt('./data/paper/fixposition/detection/outp_sonyg1.txt',usecols=(2,0,1))[:rows].T
# detect_3 = np.loadtxt('./data/paper/fixposition/detection/outp_sonyalpha5001.txt',usecols=(2,0,1))[:rows].T
# detect_4 = np.loadtxt('./data/paper/fixposition/detection/outp_mate7_1.txt',usecols=(2,0,1))[:rows].T
# detect_5 = np.loadtxt('./data/paper/fixposition/detection/outp_gopro1.txt',usecols=(2,0,1))[:rows].T
# detect_6 = np.loadtxt('./data/paper/fixposition/detection/outp_sony5n1.txt',usecols=(2,0,1))[:rows].T

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
flight.beta = np.array([0, 465.2250, -406.2928, -452.2184, 546.9844, 248.3173])
#flight.beta_rs = np.array([0,0,0,0,0,0])

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
    res = flight.BA(cam_temp,rs_crt=rs_crt)

    print('\nMean error of each camera after BA:    ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

    flight.remove_outliers(flight.sequence[:cam_temp],thres=outlier_thres)

    res = flight.BA(cam_temp,rs_crt=rs_crt)

    print('\nMean error of each camera after second BA:    ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

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
#vis.show_trajectory_3D(flight.traj[1:],line=False)

#with open('data2/cvpr_res/fixposition/trajectory/flight_spline_d_3.pkl','wb') as f:
#     pickle.dump(flight, f)

with open('data2/cvpr_res/fixposition/trajectory/flight_'+str(rows)+'_jac_'+str(sparse)+'_rs_'+str(rs_crt)+'_cam_mod_'+str(cam_model)+'.pkl','wb') as file:
    pickle.dump(flight, file)


print('Finish !')