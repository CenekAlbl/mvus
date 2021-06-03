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

import cv2
from reconstruction import epipolar as ep
import argparse
import os

# parse the input
a = argparse.ArgumentParser()
a.add_argument("--config_file", type=str, help="path to the proper config file", required=True)
a.add_argument("--debug", action="store_true", help="debug mode: run with ground truth)")

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

# # Initialize with the static part
# flight.init_static(inlier_only=True, debug=args.debug)

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
        rs_bounds=flight.settings['rs_bounds'],debug=args.debug)

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
        rs_bounds=flight.settings['rs_bounds'],debug=args.debug)

    print('\nMean error of each camera after the second BA:   ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))
    print('\nMean error of the static part in each camera after the second BA:    ', np.asarray([np.mean(flight.error_cam_static(x, debug=args.debug)) for x in flight.sequence[:cam_temp]]))

    num_end = flight.numCam if flight.find_order else len(flight.sequence)
    if cam_temp == num_end:
        print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
        break

    # Add the next camera and get its pose
    flight.get_camera_pose(flight.sequence[cam_temp], flight.sequence[:cam_temp], debug=args.debug)

    # Triangulate new points and update the 3D spline
    flight.triangulate(flight.sequence[cam_temp], flight.sequence[:cam_temp], thres=flight.settings['thres_triangulation'], factor_t2s=flight.settings['smooth_factor'], factor_s2t=flight.settings['sampling_rate'])
    # Triangulate new points and update the static scene
    flight.triangulate_static(flight.sequence[cam_temp], flight.sequence[:cam_temp], thres=flight.settings['thres_triangulation'])

    print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
    cam_temp += 1
    flight.traj_len = []

# Discretize trajectory
flight.spline_to_traj(sampling_rate=1)
# save the 2d trajectories
if 'save_2d' in flight.settings.keys() and flight.settings['save_2d']:
    for i, cam in enumerate(flight.cameras):
        x_res = cam.dist_point3d(flight.traj[1:])
        x_ori = flight.detections[i][1:]
        # visualize the reprojection of the reconstructed trajectories
        vis.show_2D_all(x_ori, x_res, title='cam'+str(i)+' trajectories', color=True, line=False, bg=cam.img)

        # # align with the raw detection
        # _ =  align_detections(flight, visualize=True)

        # save the reprojected trajectories
        traj_res = np.vstack([x_res, flight.traj[0]]).T
        # save the raw detection (replace the timestamp to the global timestamp)
        det_ori_global = np.vstack([x_ori, flight.detections_global[i][0]]).T
        np.savetxt(flight.settings['save_2d_path'].replace('.txt', '_cam'+str(i)+'.txt'), traj_res, delimiter=' ')
        np.savetxt(flight.settings['save_2d_path'].replace('.txt', '_det_ori_global_cam'+str(i)+'.txt'), det_ori_global, delimiter=' ')

# Visualize the 3D trajectory
vis.show_trajectory_3D(flight.traj[1:],line=False)

# Align with the ground truth static points if available
if flight.gt_static is not None:
    # Transform the ground truth static 3d points
    static_ref = align_gt_static(flight)
    # Visualize the reconstructed 3D static points and the ground truth static points
    vis.show_3D_all(static_ref, flight.static[:, flight.inlier_mask > 0], color=True, line=False)
    for i, cam in enumerate(flight.cameras):
        # x_res = cam.projectPoint(flight.static[:, cam.index_2d_3d])[:-1]
        x_res = cam.dist_point3d(flight.static[:, cam.index_2d_3d])
        x_ori = cam.get_gt_pts()
        vis.show_2D_all(x_ori, x_res, title='cam'+str(i), color=True, line=False, bg=cam.img)
else:
    # Visualize the 3D static points
    vis.show_3D_all(flight.static[:, flight.inlier_mask > 0], color=False, line=False)
    # no ground truth exists, plot the reprojection in 2d
    for i, cam in enumerate(flight.cameras):
        x_res = cam.dist_point3d(flight.static[:, cam.index_2d_3d])
        # x_res = cam.projectPoints(flight.static[:, cam.index_2d_3d])[:-1]
        if args.debug:
            x_ori = cam.get_gt_pts()
        else:
            x_ori = cam.get_points()
        vis.show_2D_all(x_ori, x_res, title='cam'+str(i), color=True, line=False, bg=cam.img)

# Align with the ground truth data if available
if len(flight.gt) > 0:
    flight.out = align_gt(flight, flight.gt['frequency'], flight.gt['filepath'], visualize=False)
with open(flight.settings['path_output'],'wb') as f:
    # unpack sift features if used
    if flight.settings['feature_extractor'] == 'sift':
        for cam in flight.cameras:
            cam.unpack_sift_kp()
    pickle.dump(flight, f)

print('Finished!')