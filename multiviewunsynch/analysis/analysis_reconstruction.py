import numpy as np
import pickle
import tools.visualization as vis
from datetime import datetime
from analysis.compare_gt import align_gt
from reconstruction import synchronization as sync

from reconstruction import epipolar as ep
from tools.util import match_overlap
from itertools import combinations
from matplotlib import pyplot as plt
import os

def reproject_ground_truth(cameras, gt_pts, gt_dets, n_bins=10, output_dir='', prefix='', flight=None):
    '''
    Function:
        Compute and plot the reprojection errors of the ground truth matches
    Input:
        cameras = the list of Camera objects to be evaluated
        gt_pts = the ground truth matches
        n_bins = number of bins for plotting the histogram of the errors
    '''
    
    combos = combinations(range(len(cameras)),2)

    repro_errors = []
    # for every pair of cameras, triangulate 3d points and reproject
    for t1, t2 in combos:
        print("triangulate camera pairs: (%d, %d)" %(t1, t2))
        # undistort gt_points
        gt_un1 = cameras[t1].undist_point(gt_pts[t1].T)
        gt_un2 = cameras[t2].undist_point(gt_pts[t2].T)

        # triangulate
        X_gt = ep.triangulate_matlab(gt_un1, gt_un2, cameras[t1].P, cameras[t2].P)

        # backproject the triangulated points on the images
        gt_repro1 = cameras[t1].dist_point3d(X_gt[:-1].T)
        gt_repro2 = cameras[t2].dist_point3d(X_gt[:-1].T)
        # compute the error
        repro_err1 = ep.reprojection_error(gt_un1, cameras[t1].projectPoint(X_gt))
        repro_err2 = ep.reprojection_error(gt_un2, cameras[t2].projectPoint(X_gt))
        repro_errors.append([repro_err1, repro_err2])
        print("mean reprojection error of camera pair (%d, %d) is %f and %f" %(t1, t2, np.mean(repro_err1), np.mean(repro_err2)))

        # histogram of reprojection error
        fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)
        axs[0].hist(repro_err1, bins=n_bins)
        axs[0].set_title('cam'+str(t1))
        axs[1].hist(repro_err2, bins=n_bins)
        axs[1].set_title('cam'+str(t2))
        plt.savefig(output_dir+prefix+'repro_cam{}_{}.png'.format(t1, t2))

        plt.show()

        # evaluate for the dynamic part
        # match between detections
        if cameras[t1].fps > cameras[t2].fps:
            det1, det2 = match_overlap(gt_dets[t1], gt_dets[t2])
        else:
            det2, det1 = match_overlap(gt_dets[t2], gt_dets[t1])

        vis.draw_detection_matches(cameras[t1].img, det1, cameras[t2].img, det2, title=prefix+'matched_detections.png', output_dir=output_dir)
        
        # undistort points
        det_un1 = cameras[t1].undist_point(det1[1:])
        det_un2 = cameras[t2].undist_point(det2[1:])
        # triangulate the detections
        Traj_gt = ep.triangulate_matlab(det_un1, det_un2, cameras[t1].P, cameras[t2].P)
        # backproject triangulated detections on the images
        traj_repro1 = cameras[t1].dist_point3d(Traj_gt[:-1].T)
        traj_repro2 = cameras[t2].dist_point3d(Traj_gt[:-1].T)
        vis.draw_detection_matches(cameras[t1].img, np.vstack([det1[0],traj_repro1]), cameras[t2].img, np.vstack([det2[0],traj_repro2]), title=prefix+'reprojected_detections.png', output_dir=output_dir)
        
        # compute the reprojection error
        traj_err1 = ep.reprojection_error(det_un1, cameras[t1].projectPoint(Traj_gt))
        traj_err2 = ep.reprojection_error(det_un2, cameras[t2].projectPoint(Traj_gt))
        repro_errors[-1] += [traj_err1, traj_err2]
        print("mean reprojection error of the trajectories of camera pair (%d, %d) is %f and %f" %(t1, t2, np.mean(traj_err1), np.mean(traj_err2)))
        # plot the histogram the reprojecton error of the dynamic part
        fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)
        axs[0].hist(traj_err1, bins=n_bins)
        axs[0].set_title('cam'+str(t1))
        axs[1].hist(traj_err2, bins=n_bins)
        axs[1].set_title('cam'+str(t2))
        plt.savefig(output_dir+prefix+'repro_traj_cam{}_{}.png'.format(t1, t2))

        plt.show()
        
        # plot images
        vis.show_2D_all(gt_pts[t1].T, gt_repro1, det1[1:], traj_repro1, title=prefix+'cam'+str(t1)+' ground truth reprojection', color=True, line=False, bg=cameras[t1].img, output_dir=output_dir)
        vis.show_2D_all(gt_pts[t2].T, gt_repro2, det2[1:], traj_repro2, title=prefix+'cam'+str(t2)+' ground truth reprojection', color=True, line=False, bg=cameras[t2].img, output_dir=output_dir)

        # plot 3D reconstructe scene
        vis.show_3D_all(X_gt, np.empty([3,0]), Traj_gt, np.empty([3,0]), color=False, line=False, flight=flight, output_dir=output_dir+prefix)

def convert_timestamps(gt_dets, alphas, betas):
    '''
    Function:
        convert the detection timestamps into the global timestamps according to the computed alpha, beta
    Input:
        cameras = the list of cameras to be evaluated
        gt_dets = ground truth detections
    Output:
        gt_dets_global = the detection pairs in the global time frame
    '''
    for gt_det, alpha, beta in zip(gt_dets, alphas, betas):
        gt_det[0] = alpha * gt_det[0] + beta

    return gt_dets

def main():
    # Output dir
    output_dir = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/eval_res_calibrated_10/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load ground truth
    gt_static_file = ['../experiments/croatia_set3/static_gt/static_cam0.txt','../experiments/croatia_set3/static_gt/static_cam1.txt']
    gt_dynamic_file = ['../experiments/croatia_set3/det_gt/cam0_3_det_gt.txt','../experiments/croatia_set3/det_gt/cam1_3_det_gt.txt']
    
    # gt_static_file = ['/scratch2/wuti/Repos/3D-Object-Trajectory-Reconstruction-Webcam/multiviewunsynch/webcam-datasets/nyc-datasets/static_gt/static_cam1_undistort_div.txt', '/scratch2/wuti/Repos/3D-Object-Trajectory-Reconstruction-Webcam/multiviewunsynch/webcam-datasets/nyc-datasets/static_gt/static_cam2_undistort_div.txt']
    # gt_dynamic_file = ['/scratch2/wuti/Repos/3D-Object-Trajectory-Reconstruction-Webcam/multiviewunsynch/webcam-datasets/nyc-datasets/20210413_1000_3min/det_opencv/set12/obj1/cam0_w1574280981_12_CSRT_obj0_undistort_div.txt', '/scratch2/wuti/Repos/3D-Object-Trajectory-Reconstruction-Webcam/multiviewunsynch/webcam-datasets/nyc-datasets/20210413_1000_3min/det_opencv/set12/obj1/cam2_w1587769795_12_CSRT_obj0_undistort_div.txt']

    gt_static = []
    for gfs in gt_static_file:
        gt_static.append(np.loadtxt(gfs, delimiter=' '))
    
    gt_dynamic = []
    for gfd in gt_dynamic_file:
        gt_dynamic.append(np.loadtxt(gfd, usecols=(2,0,1), delimiter=' ').T)

    # Load scenes
    # COLMAP GUESS
    # data_file_dynamic = '/scratch2/wuti/Repos/mvus/experiments/dynamic/nyc_colmap_dynamic_ori_inlier_30.pkl'
    # data_file_static = '/scratch2/wuti/Repos/mvus/experiments/static/nyc_colmap_static_superglue_colmap2_30.pkl'
    # data_file_static_dynamic = '/scratch2/wuti/Repos/mvus/experiments/static_dynamic/nyc_colmap_static_dynamic_superglue_colmap2_30.pkl'
    # data_file_static_then_dynamic = '/scratch2/wuti/Repos/mvus/experiments/static_then_dynamic/nyc_colmap_static_then_dynamic_superglue_30.pkl'
    
    # COLMAP GUESS 0
    # data_file_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/calibrated_dynamic/nyc_colmap_dynamic_ori_inlier_calibrated_30.pkl'
    # data_file_static = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/colmap2_static/nyc_colmap_static_superglue_colmap2_30.pkl'
    # data_file_static_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/colmap2_static_dynamic/nyc_colmap_static_dynamic_superglue_colmap2_30.pkl'

    # COLMAP GUESS 0
    # data_file_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/calibrated_dynamic/nyc_colmap_dynamic_ori_inlier_calibrated_30.pkl'
    data_file_static = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/colmap3_static/nyc_colmap_static_superglue_colmap2_30.pkl'
    data_file_static_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/colmap3_static_dynamic/nyc_colmap_static_dynamic_superglue_colmap2_30.pkl'

    # CALIBRATED
    # data_file_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/calibrated_dynamic/nyc_colmap_dynamic_ori_inlier_calibrated_30.pkl'
    # data_file_static = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/calibrated_static/nyc_colmap_static_superglue_calibrated_30.pkl'
    # data_file_static_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/calibrated_static_dynamic/nyc_colmap_static_dynamic_superglue_calibrated_30.pkl'

    # CALIBRATED F10
    data_file_static = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/calibrated_static_10/nyc_colmap_static_superglue_calibrated_10.pkl'
    data_file_static_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/calibrated_static_dynamic_10/nyc_colmap_static_dynamic_superglue_calibrated_10.pkl'


    # UNDISTORT
    # data_file_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/undistort_dynamic/nyc_colmap_dynamic_ori_inlier_undistort_30.pkl'
    # data_file_static = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/undistort_static/nyc_colmap_static_superglue_undistort_30.pkl'
    # data_file_static_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/undistort_static_dynamic/nyc_colmap_static_dynamic_superglue_undistort_30.pkl'
    # data_file_static_then_dynamic = '/scratch2/wuti/Repos/mvus/experiments/static_then_dynamic/nyc_colmap_static_then_dynamic_superglue_30.pkl'
    
    # with open(data_file_dynamic, 'rb') as file:
    #     flight_dynamic = pickle.load(file)

    with open(data_file_static, 'rb') as file:
        flight_static = pickle.load(file)

    with open(data_file_static_dynamic, 'rb') as file:
        flight_static_dynamic = pickle.load(file)

    # with open(data_file_static_then_dynamic, 'rb') as file:
    #     flight_static_then_dynamic = pickle.load(file)

    # Analysis
    # 2D reprojection error
    print("Plot reprojection error")
    
    # # dynamic only
    # print("Reconstructions from dynamic-only setting")
    # # convert dynamic part timestamp
    # gt_dynamic1 = [gt_dynamic[x].copy() for x in flight_dynamic.sequence]
    # gt_dynamic1 = convert_timestamps(gt_dynamic1, flight_dynamic.alpha, flight_dynamic.beta)
    # reproject_ground_truth(flight_dynamic.cameras, gt_static, gt_dynamic1, output_dir=output_dir, prefix='dynamic_only_', flight=flight_dynamic)
    
    print('\n#################################################################\n')
    print("Reconstructions from static-only setting")
    gt_dynamic2 = [gt_dynamic[x].copy() for x in flight_static_dynamic.sequence]
    # no optimization for synchronization, use fps to convert timestamps
    for i, cam in enumerate(flight_static.cameras):
        gt_dynamic2[i][0] *= flight_static.cameras[flight_static.ref_cam].fps/cam.fps
    reproject_ground_truth(flight_static.cameras, gt_static, gt_dynamic2, output_dir=output_dir, prefix='static_only_', flight=flight_static)

    print('\n#################################################################\n')
    print("Reconstructions from static-dynamic setting")
    gt_dynamic3 = [gt_dynamic[x].copy() for x in flight_static_dynamic.sequence]
    gt_dynamic3 = convert_timestamps(gt_dynamic3, flight_static_dynamic.alpha, flight_static_dynamic.beta)
    reproject_ground_truth(flight_static_dynamic.cameras, gt_static, gt_dynamic3, output_dir=output_dir, prefix='static_dynamic_', flight=flight_static_dynamic)

    print('\n#################################################################\n')
    print("Reconstructions from static-only setting")
    reproject_ground_truth(flight_static.cameras, gt_static, gt_dynamic3, output_dir=output_dir, prefix='static_only_sync_', flight=flight_static)

    # print('\n#################################################################\n')
    # print("Reconstructions from static-then-dynamic setting")
    # gt_dynamic4 = [gt_dynamic[x].copy() for x in flight_static_then_dynamic.sequence]
    # gt_dynamic4 = convert_timestamps(gt_dynamic4, flight_static_then_dynamic.alpha, flight_static_then_dynamic.beta)
    # reproject_ground_truth(flight_static_then_dynamic.cameras, gt_static, gt_dynamic4, output_dir=output_dir, prefix='static_then_dynamic_', flight=flight_static_then_dynamic)

    print('Finish!')

if __name__ == '__main__':
    main()