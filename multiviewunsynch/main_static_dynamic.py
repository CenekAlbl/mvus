# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pickle

from numpy.linalg.linalg import det
from tools import visualization as vis
from datetime import datetime
from reconstruction import common
from analysis.compare_gt import align_gt, align_gt_static, align_detections
import sys
from tools.util import unpack_sift_kp

import cv2
from reconstruction import epipolar as ep
import argparse
import os

# parse the input
a = argparse.ArgumentParser()
a.add_argument("--config_file", type=str, help="path to the proper config file", required=True)
a.add_argument("--debug", action="store_true", help="debug mode: run with ground truth)")
a.add_argument("--scale", action="store_true", help="scale variable in BA")

args = a.parse_args()

print('Reconstruct with both static part and dynamic part of the scene.\n')

if args.debug:
    print("RUN ON DEBUG MODE WITH GROUND TRUTH STATIC POINTS")

# Initialize a scene from the json template
flight = common.create_scene(args.config_file)

# Truncate detections
flight.cut_detection(second=flight.settings['cut_detection_second'])

# Add prior alpha
flight.init_alpha()

# Compute time shift for each camera
flight.time_shift()

# Convert raw detections into the global timeline
flight.detection_to_global()

# Initialize the first 3D trajectory
flight.init_traj(error=flight.settings['thres_Fmatix'], inlier_only=True, debug=args.debug)

# Convert discrete trajectory to spline representation
flight.traj_to_spline(smooth_factor=flight.settings['smooth_factor'])

'''---------------Incremental reconstruction----------------'''
start = datetime.now()
np.set_printoptions(precision=4)

cam_temp = 2
while True:
    print('\n----------------- Bundle Adjustment with {} cameras -----------------'.format(cam_temp))
    print('\nMean error of each camera before BA:   ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))
    print('\nMean error of the static part in each camera before BA:   ', np.asarray([np.mean(flight.error_cam_static(x, debug=args.debug)) for x in flight.sequence[:cam_temp]]))

    print('\nDoing the first BA')
    # Bundle adjustment
    # res = flight.BA_static(cam_temp, debug=args.debug)
    res = flight.BA(cam_temp, rs=flight.settings['rolling_shutter'],\
        motion_reg=flight.settings['motion_reg'],\
        motion_weights=flight.settings['motion_weights'],\
        rs_bounds=flight.settings['rs_bounds'],debug=args.debug,scaling=args.scale)

    print('\nMean error of each camera after the first BA:   ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))
    print('\nMean error of the static part in each camera after the first BA:    ', np.asarray([np.mean(flight.error_cam_static(x, debug=args.debug)) for x in flight.sequence[:cam_temp]]))

    # remove outliers
    print('\nRemove outliers after first BA')
    # flight.remove_outliers_static(flight.sequence[:cam_temp], thres=flight.settings['thres_outlier'], verbose=True, debug=args.debug)
    flight.remove_outliers(flight.sequence[:cam_temp], thres=flight.settings['thres_outlier'], verbose=True, debug=args.debug)
    
    print('\nDoing the second BA')
    # Bundle adjustment after outlier removal
    # res = flight.BA_static(cam_temp, debug=args.debug)
    res = flight.BA(cam_temp, rs=flight.settings['rolling_shutter'],\
        motion_reg=flight.settings['motion_reg'],\
        motion_weights=flight.settings['motion_weights'],\
        rs_bounds=flight.settings['rs_bounds'],debug=args.debug, scaling=args.scale)

    print('\nMean error of each camera after the second BA:   ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))
    print('\nMean error of the static part in each camera after the second BA:    ', np.asarray([np.mean(flight.error_cam_static(x, debug=args.debug)) for x in flight.sequence[:cam_temp]]))

    num_end = flight.numCam if flight.find_order else len(flight.sequence)
    if cam_temp == num_end:
        print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
        break

    # Select the next camera if not pre-defined
    flight.select_most_overlap()
    
    # Add the next camera and get its pose
    flight.get_camera_pose(flight.sequence[cam_temp], debug=args.debug)

    # Triangulate new points and update the 3D spline
    flight.triangulate(flight.sequence[cam_temp], flight.sequence[:cam_temp], thres=flight.settings['thres_triangulation'], factor_t2s=flight.settings['smooth_factor'], factor_s2t=flight.settings['sampling_rate'])
    # Triangulate new points and update the static scene
    flight.triangulate_static(flight.sequence[cam_temp], flight.sequence[:cam_temp], thres=flight.settings['thres_triangulation'])

    print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
    cam_temp += 1
    flight.traj_len = []

# Discretize trajectory
flight.spline_to_traj(sampling_rate=1)

# Visualize the 3D trajectory
vis.show_trajectory_3D(flight.traj[1:],line=False)

# Align with the ground truth static points if available
if flight.gt_static is not None:
    # Transform the ground truth static 3d points
    static_ref = align_gt_static(flight)
    # Visualize the reconstructed 3D static points and the ground truth static points
    vis.show_3D_all(static_ref, flight.static[:, flight.inlier_mask > 0], color=True, line=False, flight=flight)
    for i, cam in enumerate(flight.cameras):
        # x_res = cam.projectPoint(flight.static[:, cam.index_2d_3d])[:-1]
        x_res = cam.dist_point3d(flight.static[:, cam.index_2d_3d])
        x_ori = cam.get_gt_pts()
        x_res_traj = cam.dist_point3d(flight.traj[1:])
        vis.show_2D_all(x_ori, x_res, flight.detections[i][1:], x_res_traj, title='cam'+str(i), color=True, line=False, bg=cam.img, label=['ground truth features', 'reconstructed ground truth features', 'extracted dynamic features', 'reconstructed trajectories'])
else:
    # Visualize the 3D static points
    vis.show_3D_all(flight.static[:, flight.inlier_mask > 0], color=False, line=False, flight=flight)
    # no ground truth exists, plot the reprojection in 2d
    for i, cam in enumerate(flight.cameras):
        x_res = cam.dist_point3d(flight.static[:, cam.index_2d_3d])
        x_res_traj = cam.dist_point3d(flight.traj[1:])
        # x_res = cam.projectPoints(flight.static[:, cam.index_2d_3d])[:-1]
        if args.debug:
            x_ori = cam.get_gt_pts()
        else:
            x_ori = cam.get_points()
        vis.show_2D_all(x_ori, x_res, flight.detections[i][1:], x_res_traj, title='cam'+str(i), color=True, line=False, bg=cam.img, label=['extracted static features', 'reconstructed static features', 'extracted dynamic features', 'reconstructed trajectories'])

# Align with the ground truth data if available
if len(flight.gt) > 0:
    flight.out = align_gt(flight, flight.gt['frequency'], flight.gt['filepath'], visualize=False)
with open(flight.settings['path_output'],'wb') as f:
    # unpack sift features if used
    if flight.settings['include_static']:
        for cam in flight.cameras:
            cam.kp = unpack_sift_kp(cam.kp)
    pickle.dump(flight, f)

print('Finished!')