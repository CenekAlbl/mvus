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
from numpy import load

'''---------------New computation----------------'''
# Setting parameters
rows = 20000
error_F = 30
cut_second = 0.5
cam_model = 10
sequence = [0,4,2,1,5,3]       #[0,2,3,4,1,5]
smooth_factor = 0.001
sampling_rate = 0.02
tri_thres = 20
rs_crt = False
sparse = True
setting = {'rows':rows, 'error_F':error_F, 'cut_second':cut_second, 'cam_model':cam_model, 'sequence':sequence, 'smooth':smooth_factor, 'sampling':sampling_rate, 'tri_thres':tri_thres}

# Motion Prior Optimization 
with open('data2/cvpr_res/fixposition/trajectory/flight_'+str(rows)+'_jac_'+str(sparse)+'_rs_'+str(rs_crt)+'_cam_mod_'+str(cam_model)+'.pkl','rb') as file:
    flight = pickle.load(file)

print('\nMean error of each camera before MP optimization:', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence]))
print('\nMedian error of each camera before MP optimization:', np.asarray([np.median(flight.error_cam(x)) for x in flight.sequence]))
print('\nMax error of each camera before MP optimization:', np.asarray([np.max(flight.error_cam(x)) for x in flight.sequence]))
#print('The mean/median/max error (distance) is {:.5f}/{:.5f}/{:.5f} meter\n'.format(np.mean(flight.error),np.median(error),np.max(error)))

time_stamps_all = np.array([])
#ts_all = np.empty([0,num_param])
section = int(40)
flight.detection_to_global()
for i in range(flight.numCam):
    flight.detections[i] = flight.detections[i][:,:5000]
    flight.detection_to_global(i)
    time_stamps_all = np.concatenate((time_stamps_all,flight.detections_global[i][0]))
    

# Interpolate 3D points for all detections within the spline interval
time_stamps_all = np.sort(time_stamps_all)

flight.spline_to_traj(t=time_stamps_all)
traj_orig = flight.traj
#

num_cams = flight.numCam
# Bundle adjustment
res = flight.BA(num_cams,max_iter=10,motion=True,motion_weights=1,rs_crt=rs_crt)

flight.traj_to_spline()
print('\nMean error of each camera after MP BA:', np.asarray([np.mean(flight.error_cam(x,motion=True)[0]) for x in flight.sequence]))

#if cam_temp == flight.numCam:
#    print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
#    break

with open('data2/cvpr_res/fixposition/trajectory/flight_mp_all_jac'+str(rows/section)+'_rs_'+str(rs_crt)+'.pkl','wb') as file:
    pickle.dump(flight, file)
# with open('./data/paper/fixposition/trajectory/flight_spline_2.pkl','wb') as f:
#     pickle.dump(flight, f)

print('Finish !')