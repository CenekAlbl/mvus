import numpy as np
import pickle
import tools.visualization as vis
from datetime import datetime
from analysis.compare_gt import align_gt
from reconstruction import synchronization as sync

from reconstruction import epipolar as ep
from tools.util import match_overlap, homogeneous
from matplotlib import pyplot as plt
import os
import cv2
import argparse
from glob import glob
import json

def reproject_ground_truth(cameras, gt_pts, gt_dets, res_json, ref_cam=0, output_dir='', prefix='', flight=None, ax=None):
    '''
    Function:
        Compute and plot the reprojection errors of the ground truth matches
    Input:
        cameras = the list of Camera objects to be evaluated
        gt_pts = the ground truth matches
        n_bins = number of bins for plotting the histogram of the errors
    '''
    f = open(os.path.join(output_dir, 'result.txt'), 'a')
    f.write(prefix)

    res = {}

    # triangulate points
    gt_pts2d = []
    traj_pts2d = []
    timestamp = gt_dets[ref_cam][0]
    Projs = []
    for cam, gt_pt, gt_det in zip(cameras, gt_pts, gt_dets):
        gt_pts2d.append(cam.undist_point(gt_pt.T))

        # sample to the ref camera
        _, mgt_det = match_overlap(gt_dets[ref_cam], gt_det)
        traj_pts2d.append(mgt_det)
        #traj_pts2d.append(np.vstack([mgt_det[0],cam.undist_point(mgt_det[1:])]))
        Projs.append(cam.P)
        timestamp = np.intersect1d(timestamp, mgt_det[0])
    
    traj_pts2d = list(map(lambda x: x[1:,np.isin(x[0], timestamp)], traj_pts2d))
    
    # stack points
    gt_pts2d = np.vstack(gt_pts2d)
    traj_pts2d = np.vstack(traj_pts2d)

    # triangulate gt
    X_gt = ep.triangulate_matlab_mv(gt_pts2d, Projs)
    Traj_gt = ep.triangulate_matlab_mv(traj_pts2d, Projs)

    # plot the reconstructed scene
    vis.show_3D_all(np.empty([3,0]), X_gt, np.empty([3,0]), Traj_gt, label=['','Reconstructed Ground Truth Matches','','Reconstructed Ground Truth Trajectory'], color=True, line=False, flight=flight, output_dir=output_dir, title=prefix+'reconstructed_scene')

    gt_err = []
    traj_err = []
    gt_labels = []
    traj_labels = []
    match_err = []
    match_labels = []
    for i, (cam, gt_pt, gt_det) in enumerate(zip(cameras, gt_pts, gt_dets)):
        # log K and d after BA
        f.write(f'\ncamera {i} K:\n {np.array2string(cam.K)}')
        f.write(f'\ncamera {i} d:\n {np.array2string(cam.d)}')
        cam_res = {}
        cam_res['K'] = cam.K.tolist()
        cam_res['d'] = cam.d.tolist()

        # reproject to image -- static matches
        pts = cam.undist_point(cam.kp[cam.index_registered_2d,:].T)
        err = ep.reprojection_error(pts, cam.projectPoint(flight.static[:,flight.inlier_mask==1]))
        match_err.append(err)
        match_labels.append('{}cam{}'.format(prefix,i))
        print("mean reprojection error of static features in camera %d is %f" %(i, np.mean(err)))
        f.write(f'\n\nmean reprojection error of static features in camera {i} is {np.mean(err)}')
        cam_res['err_static'] = np.mean(err)

        # reproject to image -- ground truth static matches
        gt_repro = cam.dist_point3d(X_gt[:-1].T)
        err1 = ep.reprojection_error(gt_pts2d[2*i:2*i+2], cam.projectPoint(X_gt))
        gt_err.append(err1)
        gt_labels.append('{}cam{}'.format(prefix,i))
        print("mean reprojection error of the ground truth matches in camera %d is %f" %(i, np.mean(err1)))
        f.write(f'\n\nmean reprojection error of the ground truth matches in camera {i} is {np.mean(err1)}')
        cam_res['err_cp'] = np.mean(err1)

        # reproject trajectories -- detection matches
        # if static-dynamic -- use the fitted trajectory
        # if len(flight.traj) > 0:
        #     traj_repro = cam.dist_point3d(flight.traj[1:])
        #     err2 = flight.error_cam(i,mode='dist')
        # else:
        traj_repro = cam.dist_point3d(Traj_gt[:-1].T)
        err2 = ep.reprojection_error(traj_pts2d[2*i:2*i+2], cam.projectPoint(Traj_gt))
        traj_err.append(err2)
        traj_labels.append('{}cam{}'.format(prefix,i))
        print("mean reprojection error of the dynamic features in camera %d is %f" %(i, np.mean(err2)))
        f.write(f'\n\nmean reprojection error of the dynamic features in camera {i} is {np.mean(err2)}')
        cam_res['err_traj'] = np.mean(err2)

        # compute the camera center
        cam_centers = []
        for j, cam_other in enumerate(cameras):
            if i == j:
                continue
            cam_centers.append(-cam_other.R.T @ cam_other.t.reshape(-1,1))
        
        cam_centers = np.hstack(cam_centers)
        cam_centers = cam.dist_point3d(cam_centers.T)

        h, w, _ = cam.img.shape
        cam_centers = cam_centers[:,(cam_centers[0] >=0) & (cam_centers[0] <= w) & (cam_centers[1] >= 0) & (cam_centers[1] <= h)]

        # plot reprojected points on the image
        vis.show_2D_all(gt_pt.T, gt_repro, cam.dist_point2d(traj_pts2d[2*i:2*i+2]), traj_repro, vec=1, title=prefix+'cam'+str(i)+'_ground_truth_reprojection', color=True, line=False, bg=cam.img, output_dir=output_dir, label=['ground truth matches', 'reconstructed ground truth matches', 'extracted dynamic features', 'reconstructed trajectories'], cam_center=cam_centers)

        res[f'cam{i}'] = cam_res
    # plot the reprojection error boxplot
    # vis.error_boxplot(gt_err, gt_labels, title=prefix+'reprojection_error_static_ground_truth.png', ax=ax[0])
    # vis.error_boxplot(traj_err, traj_labels, title=prefix+'reprojection_error_dynamic.png', ax=ax[1])
    # vis.error_boxplot(match_err, match_labels, title=prefix+'reprojection_error_static.png', ax=ax[2])

    f.write('\n\n#################################################################\n\n')
    f.close()

    res_json[prefix[:-1]] = res

    return gt_err, gt_labels, traj_err, traj_labels, match_err, match_labels

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

def undistort_image(cam, output_dir = '', title=None):
    HEIGHT, WIDTH, _ = cam.img.shape
    newK, roi = cv2.getOptimalNewCameraMatrix(cam.K, cam.d, (WIDTH, HEIGHT), 1, (WIDTH, HEIGHT))

    dst = cv2.undistort(cam.img, cam.K, cam.d, None, newK)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    if title is not None:
        cv2.imwrite(os.path.join(output_dir, title), dst)
    else:
        cv2.imwrite(os.path.join(output_dir, 'undistorted.png'), dst)


def main(data_file_static, data_file_static_dynamic, data_file_static_dynamic_no_sync, gt_static_file, output_dir):
    res_json = {}

    gt_static = []
    for gfs in gt_static_file:
        gt_static.append(np.loadtxt(gfs, delimiter=' '))
    
    # read result files
    with open(data_file_static, 'rb') as file:
        flight_static = pickle.load(file)

    with open(data_file_static_dynamic, 'rb') as file:
        flight_static_dynamic = pickle.load(file)

    with open(data_file_static_dynamic_no_sync, 'rb') as file:
        flight_static_dynamic_no_sync = pickle.load(file)

    # print("Undistort images")
    # for i, cam in enumerate(flight_static.cameras):
    #     undistort_image(cam, output_dir=output_dir, title='static_only_cam{}.png'.format(i))
    # for i, cam in enumerate(flight_static_dynamic.cameras):
    #     undistort_image(cam, output_dir=output_dir, title='static_dynamic_sync_cam{}.png'.format(i))
    # for i, cam in enumerate(flight_static_dynamic_no_sync.cameras):
    #     undistort_image(cam, output_dir=output_dir, title='static_dynamic_unsync_cam{}.png'.format(i))

    # Analysis
    # 2D reprojection error
    print("Plot reprojection error")
    
    _, axs1 = plt.subplots(3, 1, sharey=True, tight_layout=True)
    _, axs2 = plt.subplots(3, 1, sharey=True, tight_layout=True)
    _, axs3 = plt.subplots(3, 1, sharey=True, tight_layout=True)

    print('\n#################################################################\n')
    print("Reconstructions from static-only setting")

    # add detections to static-only
    flight_static.detections = flight_static_dynamic_no_sync.detections
    # flight_static.detections = [gt_dynamic[x].copy() for x in flight_static.sequence]
    flight_static.settings['cf_exact'] = True
    flight_static.cut_detection(second=flight_static.settings['cut_detection_second'])
    flight_static.init_alpha()
    flight_static.time_shift()
    # Convert raw detections into the global timeline
    flight_static.detection_to_global()
    gt_dynamic2 = [flight_static.detections_global[x] for x in flight_static.sequence]
    gt_err1, gt_label1, traj_err1, traj_label1, match_err1, match_label1 = reproject_ground_truth(flight_static.cameras, gt_static, gt_dynamic2, res_json,ref_cam=flight_static_dynamic.ref_cam, output_dir=output_dir, prefix='static_only_', flight=flight_static, ax=[axs1[0],axs2[0],axs3[0]])
    plt.show()
    print('\n#################################################################\n')
    print("Reconstructions from static-dynamic setting")
    gt_dynamic3 = [flight_static_dynamic.detections[x].copy() for x in flight_static_dynamic.sequence]    
    flight_static_dynamic.detections = flight_static_dynamic_no_sync.detections
    # flight_static_dynamic.detections = [gt_dynamic[x].copy() for x in flight_static_dynamic.sequence]
    flight_static_dynamic.detection_to_global()
    gt_dynamic3 = [flight_static_dynamic.detections_global[x] for x in flight_static_dynamic.sequence]
    gt_err2, gt_label2, traj_err2, traj_label2, match_err2, match_label2 = reproject_ground_truth(flight_static_dynamic.cameras, gt_static, gt_dynamic3, res_json,ref_cam=flight_static_dynamic.ref_cam, output_dir=output_dir, prefix='static_dynamic_sync_', flight=flight_static_dynamic, ax=[axs1[1],axs2[1],axs3[1]])
    plt.show()
    print('\n#################################################################\n')
    print("Reconstructions from static-dynamic-no-sync setting")
    # flight_static_dynamic_no_sync.detections = [gt_dynamic[x].copy() for x in flight_static_dynamic_no_sync.sequence]
    flight_static_dynamic_no_sync.detection_to_global()
    gt_dynamic4 = [flight_static_dynamic_no_sync.detections_global[x] for x in flight_static_dynamic_no_sync.sequence]
    gt_err3, gt_label3, traj_err3, traj_label3, match_err3, match_label3 = reproject_ground_truth(flight_static_dynamic_no_sync.cameras, gt_static, gt_dynamic4, res_json,ref_cam=flight_static_dynamic_no_sync.ref_cam, output_dir=output_dir, prefix='static_dynamic_unsync_', flight=flight_static_dynamic_no_sync, ax=[axs1[2],axs2[2],axs3[2]])

    # save result
    with open(os.path.join(output_dir, 'result.json'), 'w') as fj:
        json.dump(res_json, fj)
    print('\n\n#################################################################\n')
    print("Visualization")

    print('\tVisualize reprojection box plot')
    # plot reprojection errors
    # reorder error terms
    gt_repro_errs = gt_err1+gt_err3+gt_err2
    gt_repro_labels = gt_label1+gt_label3+gt_label2
    gt_repro_errs = gt_repro_errs[::2] + gt_repro_errs[1:][::2]
    gt_repro_labels = gt_repro_labels[::2] + gt_repro_labels[1:][::2]

    match_repro_errs = match_err1+match_err3+match_err2
    match_repro_labels = match_label1+match_label3+match_label2
    match_repro_errs = match_repro_errs[::2] + match_repro_errs[1:][::2]
    match_repro_labels = match_repro_labels[::2] + match_repro_labels[1:][::2]

    traj_repro_errs = traj_err1+traj_err3+traj_err2
    traj_repro_labels = traj_label1+traj_label3+traj_label2
    traj_repro_errs = traj_repro_errs[::2] + traj_repro_errs[1:][::2]
    traj_repro_labels = traj_repro_labels[::2] + traj_repro_labels[1:][::2]

    vis.error_boxplot(gt_repro_errs, gt_repro_labels, title='reprojection_error_static_ground_truth.png', output_dir=output_dir)
    # plt.show()
    vis.error_boxplot(traj_repro_errs, traj_repro_labels, title='reprojection_error_dynamic.png', output_dir=output_dir)
    # plt.show()
    vis.error_boxplot(match_repro_errs, match_repro_labels, title='reprojection_error_static.png', output_dir=output_dir)
    # plt.show()

    traj_repro_errs2 = traj_err2+traj_err3
    traj_repro_labels2 = traj_label2+traj_label3
    traj_repro_errs2 = traj_repro_errs2[::2] + traj_repro_errs2[1:][::2]
    traj_repro_labels2 = traj_repro_labels2[::2] + traj_repro_labels2[1:][::2]
    vis.error_boxplot(traj_repro_errs2, traj_repro_labels2, title='reprojection_error_dynamic2.png', output_dir=output_dir)
    # plt.show()

    print('\tVisualize reprojection histogram')
    # plot error histograms
    width = 1
    fig_gt_hist, gt_hist_axs = plt.subplots(2, 1, sharex=True, sharey=True, dpi=300)
    xlim = np.max(np.hstack(gt_repro_errs))
    for i, (cam_err1, cam_label1, cam_err2, cam_label2, cam_err3, cam_label3) in enumerate(zip(gt_err1, gt_label1, gt_err2, gt_label2, gt_err2, gt_label3)):
        vis.error_histogram(cam_err1, cam_err2, cam_err3, labels=[cam_label1, cam_label2, cam_label3],ax=gt_hist_axs[i],title='cam '+str(i),xlim=80, num_bins=20, bin_width=width)
    xticks = np.arange(0,90,10)
    l2 = ["{:.0f}".format(x) for x in xticks]
    loc2 = xticks
    l2[-1] += '+'
    gt_hist_axs[0].set_xticks(loc2)
    gt_hist_axs[1].set_xticks(loc2)
    gt_hist_axs[1].set_xticklabels(l2)
    fig_gt_hist.suptitle('reprojection_error_histogram_static_ground_truth')
    fig_gt_hist.savefig(os.path.join(output_dir,'reprojection_error_histogram_static_ground_truth.png'))
    # plt.show()

    fig_traj_hist, traj_hist_axs = plt.subplots(2, 1, sharex=True, sharey=True, dpi=300)
    xlim = np.max(np.hstack(traj_repro_errs))
    for i, (cam_err1, cam_label1, cam_err2, cam_label2, cam_err3, cam_label3) in enumerate(zip(traj_err1, traj_label1, traj_err2, traj_label2, traj_err3, traj_label3)):
        vis.error_histogram(cam_err1, cam_err2, cam_err3, labels=[cam_label1, cam_label2, cam_label3],ax=traj_hist_axs[i],title='cam '+str(i),xlim=80, num_bins=20, bin_width=width)
    traj_hist_axs[0].set_xticks(loc2)
    traj_hist_axs[1].set_xticks(loc2)
    traj_hist_axs[1].set_xticklabels(l2)
    fig_traj_hist.suptitle('reprojection_error_histogram_dynamic')
    fig_traj_hist.savefig(os.path.join(output_dir,'reprojection_error_histogram_dynamic.png'))
    # plt.show()

    fig_match_hist, match_hist_axs = plt.subplots(2, 1, sharex=True, sharey=True, dpi=300)
    xlim = np.max(np.hstack(match_repro_errs))
    for i, (cam_err1, cam_label1, cam_err2, cam_label2, cam_err3, cam_label3) in enumerate(zip(match_err1, match_label1, match_err2, match_label2, match_err3, match_label3)):
        vis.error_histogram(cam_err1, cam_err2, cam_err3, labels=[cam_label1, cam_label2, cam_label3],ax=match_hist_axs[i],title='cam '+str(i),xlim=80, num_bins=20, bin_width=width)
    match_hist_axs[0].set_xticks(loc2)
    match_hist_axs[1].set_xticks(loc2)
    match_hist_axs[1].set_xticklabels(l2)
    fig_match_hist.suptitle('reprojection_error_histogram_static')
    fig_match_hist.savefig(os.path.join(output_dir,'reprojection_error_histogram_static.png'))
    # plt.show()

    print('Finish!')

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--results', nargs='+', required=True, help='list of result files')
    a.add_argument('--gt_path', type=str, required=True, help='path to ground truth folder')
    a.add_argument('--output_dir', type=str, required=True, help='path to output folder')

    opt = a.parse_args()

    # read result files
    assert len(opt.results) == 3, 'Wrong number of result files.'
    data_file_static, data_file_static_dynamic, data_file_static_dynamic_no_sync = opt.results

    # read gt control points
    gt_static_file = sorted(glob(os.path.join(opt.gt_path, '*.txt')))
    assert len(gt_static_file) >= 2, 'Not enough control point files found.'

    # specify output directory
    output_dir = opt.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    main(data_file_static, data_file_static_dynamic, data_file_static_dynamic_no_sync, gt_static_file, output_dir)