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
import cv2

def reproject_ground_truth(cameras, gt_pts, gt_dets, ref_cam=0, output_dir='', prefix='', flight=None, ax=None):
    '''
    Function:
        Compute and plot the reprojection errors of the ground truth matches
    Input:
        cameras = the list of Camera objects to be evaluated
        gt_pts = the ground truth matches
        n_bins = number of bins for plotting the histogram of the errors
    '''

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
    vis.show_3D_all(np.empty([3,0]), X_gt, np.empty([3,0]), Traj_gt, label=['','Reconstructed Ground Truth Matches','','Reconstructed Ground Truth Trajectory'], color=True, line=False, flight=flight, output_dir=output_dir+prefix)

    gt_err = []
    traj_err = []
    gt_labels = []
    traj_labels = []
    match_err = []
    match_labels = []
    for i, (cam, gt_pt, gt_det) in enumerate(zip(cameras, gt_pts, gt_dets)):
        # reproject to image -- static matches
        pts = cam.undist_point(cam.kp[cam.index_registered_2d,:].T)
        err = ep.reprojection_error(pts, cam.projectPoint(flight.static[:,flight.inlier_mask==1]))
        match_err.append(err)
        match_labels.append('{} cam{}'.format(prefix,i))
        print("mean reprojection error of static features in camera %d is %f" %(i, np.mean(err)))

        # reproject to image -- ground truth static matches
        gt_repro = cam.dist_point3d(X_gt[:-1].T)
        err1 = ep.reprojection_error(gt_pts2d[2*i:2*i+2], cam.projectPoint(X_gt))
        gt_err.append(err1)
        gt_labels.append('{} cam{}'.format(prefix,i))
        print("mean reprojection error of the ground truth matches in camera %d is %f" %(i, np.mean(err1)))

        # reproject trajectories -- detection matches
        # if static-dynamic -- use the fitted trajectory
        # if len(flight.traj) > 0:
        #     traj_repro = cam.dist_point3d(flight.traj[1:])
        #     err2 = flight.error_cam(i,mode='dist')
        # else:
        traj_repro = cam.dist_point3d(Traj_gt[:-1].T)
        err2 = ep.reprojection_error(traj_pts2d[2*i:2*i+2], cam.projectPoint(Traj_gt))
        traj_err.append(err2)
        traj_labels.append('{} cam{}'.format(prefix,i))
        print("mean reprojection error of the dynamic features in camera %d is %f" %(i, np.mean(err2)))

        # plot reprojected points on the image
        vis.show_2D_all(gt_pt.T, gt_repro, cam.dist_point2d(traj_pts2d[2*i:2*i+2]), traj_repro, title=prefix+'cam'+str(i)+' ground truth reprojection', color=True, line=False, bg=cam.img, output_dir=output_dir)

    # plot the reprojection error boxplot
    vis.error_boxplot(gt_err, gt_labels, title=prefix+'reprojection_error_static_ground_truth.png', ax=ax[0])
    vis.error_boxplot(traj_err, traj_labels, title=prefix+'reprojection_error_dynamic.png', ax=ax[1])
    vis.error_boxplot(match_err, match_labels, title=prefix+'reprojection_error_static.png', ax=ax[2])

    return gt_err, gt_labels, traj_err, traj_labels, match_err, match_labels

def reproject_ground_truth_2views(cameras, gt_pts, gt_dets, ref_cam=0, n_bins=10, output_dir='', prefix='', flight=None, ax=None):
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
    output_dir = '../experiments/croatia_set3/eval_res_calibrated_30/'
    # output_dir = '../experiments/croatia_set3/eval_res_undistorted_30/'

    # output_dir = '../experiments/nyc_set12/eval_res_calibrated_30/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load ground truth
    gt_static_file = ['../experiments/croatia_set3/static_gt/static_cam0.txt','../experiments/croatia_set3/static_gt/static_cam1.txt']
    gt_dynamic_file = ['../experiments/croatia_set3/det_gt/cam0_3_det_gt.txt','../experiments/croatia_set3/det_gt/cam1_3_det_gt.txt']
    
    # gt_static_file = ['../experiments/croatia_set3/static_gt_un/static_cam0_un.txt','../experiments/croatia_set3/static_gt_un/static_cam1_un.txt']
    # gt_dynamic_file = ['../experiments/croatia_set3/det_gt/cam0_3_det_gt_undistort_div.txt','../experiments/croatia_set3/det_gt/cam1_3_det_gt_undistort_div.txt']
    
    # gt_static_file = ['../experiments/nyc_set12/static_gt/static_cam0.txt','../experiments/nyc_set12/static_gt/static_cam2.txt']
    # gt_dynamic_file = ['../experiments/nyc_set12/det_gt/cam0_12.txt','../experiments/nyc_set12/det_gt/cam2_12.txt']
    
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
    # data_file_static = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/colmap3_static/nyc_colmap_static_superglue_colmap2_30.pkl'
    # data_file_static_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/colmap3_static_dynamic/nyc_colmap_static_dynamic_superglue_colmap2_30.pkl'

    # CALIBRATED
    # data_file_dynamic = '../experiments/croatia_set3/dynamic/croatia_dynamic_superglue_30.pkl'
    data_file_static = '../experiments/croatia_set3/static/croatia_static_superglue_30.pkl'
    data_file_static_dynamic = '../experiments/croatia_set3/static_dynamic/croatia_static_dynamic_superglue_30.pkl'

    # CALIBRATED F10
    # data_file_static = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/calibrated_static_10/nyc_colmap_static_superglue_calibrated_10.pkl'
    # data_file_static_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/calibrated_static_dynamic_10/nyc_colmap_static_dynamic_superglue_calibrated_10.pkl'


    # UNDISTORT
    # data_file_dynamic = '/scratch2/wuti/Repos/mvus/experiments/set12_obj0/undistort_dynamic/nyc_colmap_dynamic_ori_inlier_undistort_30.pkl'
    # data_file_dynamic = '../experiments/croatia_set3/dynamic/croatia_dynamic_superglue_30.pkl'
    # data_file_static = '../experiments/croatia_set3/static_un/croatia_static_un_superglue_30.pkl'
    # data_file_static_dynamic = '../experiments/croatia_set3/static_dynamic_un/croatia_static_dynamic_un_superglue_30.pkl'

# data_file_dynamic = '../experiments/croatia_set3/dynamic/croatia_dynamic_superglue_30.pkl'
    # data_file_static = '../experiments/nyc_set12/static/nyc_static_superglue_30.pkl'
    # data_file_static_dynamic = '../experiments/nyc_set12/static_dynamic/nyc_static_dynamic_superglue_30.pkl'


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
    
    _, axs1 = plt.subplots(3, 1, sharey=True, tight_layout=True)
    _, axs2 = plt.subplots(3, 1, sharey=True, tight_layout=True)
    _, axs3 = plt.subplots(3, 1, sharey=True, tight_layout=True)

    print('\n#################################################################\n')
    print("Reconstructions from static-only setting")
    # gt_dynamic2 = [gt_dynamic[x].copy() for x in flight_static_dynamic.sequence]
    # gt_dynamic2 = [flight_static_dynamic.detections[x].copy() for x in flight_static_dynamic.sequence]
    # # no optimization for synchronization, use fps and manually aligned time offset to convert timestamps
    # offsets = [562,25]
    # # offsets = [0,0]
    # for i, cam in enumerate(flight_static.cameras):
    #     gt_dynamic2[i][0] -= offsets[i]
    #     gt_dynamic2[i][0] *= flight_static.cameras[flight_static.ref_cam].fps/cam.fps
    flight_static.detections = flight_static_dynamic.detections
    flight_static.settings['cf_exact'] = True
    flight_static.cut_detection(second=flight_static.settings['cut_detection_second'])
    flight_static.init_alpha()
    flight_static.time_shift()
    # Convert raw detections into the global timeline
    flight_static.detection_to_global()
    gt_dynamic2 = [flight_static.detections_global[x] for x in flight_static.sequence]
    gt_err1, gt_label1, traj_err1, traj_label1, match_err1, match_label1 = reproject_ground_truth(flight_static.cameras, gt_static, gt_dynamic2,ref_cam=flight_static_dynamic.ref_cam, output_dir=output_dir, prefix='static_only_', flight=flight_static, ax=[axs1[0],axs2[0],axs3[0]])
    plt.show()
    print('\n#################################################################\n')
    print("Reconstructions from static-dynamic setting")
    # gt_dynamic3 = [gt_dynamic[x].copy() for x in flight_static_dynamic.sequence]
    # gt_dynamic3 = convert_timestamps(gt_dynamic3, flight_static_dynamic.alpha, flight_static_dynamic.beta)
    gt_dynamic3 = [flight_static_dynamic.detections[x].copy() for x in flight_static_dynamic.sequence]
    gt_dynamic3 = convert_timestamps(gt_dynamic3, flight_static_dynamic.alpha, flight_static_dynamic.beta)
    gt_dynamic3 = [flight_static_dynamic.detections_global[x] for x in flight_static_dynamic.sequence]
    gt_err2, gt_label2, traj_err2, traj_label2, match_err2, match_label2 = reproject_ground_truth(flight_static_dynamic.cameras, gt_static, gt_dynamic3,ref_cam=flight_static_dynamic.ref_cam, output_dir=output_dir, prefix='static_dynamic_', flight=flight_static_dynamic, ax=[axs1[1],axs2[1],axs3[1]])
    plt.show()
    print('\n#################################################################\n')
    print("Reconstructions from static-only setting")
    # gt_err3, gt_label3, traj_err3, traj_label3, match_err3, match_label3 = reproject_ground_truth(flight_static.cameras, gt_static, gt_dynamic3,ref_cam=flight_static_dynamic.ref_cam, output_dir=output_dir, prefix='static_only_sync_', flight=flight_static, ax=[axs1[2],axs2[2],axs3[1]])


    # plot reprojection errors
    # reorder error terms
    gt_repro_errs = gt_err1+gt_err2
    gt_repro_labels = gt_label1+gt_label2
    gt_repro_errs = gt_repro_errs[::2] + gt_repro_errs[1:][::2]
    gt_repro_labels = gt_repro_labels[::2] + gt_repro_labels[1:][::2]

    match_repro_errs = match_err1+match_err2
    match_repro_labels = match_label1+match_label2
    match_repro_errs = match_repro_errs[::2] + match_repro_errs[1:][::2]
    match_repro_labels = match_repro_labels[::2] + match_repro_labels[1:][::2]

    traj_repro_errs = traj_err1+traj_err2
    traj_repro_labels = traj_label1+traj_label2
    traj_repro_errs = traj_repro_errs[::2] + traj_repro_errs[1:][::2]
    traj_repro_labels = traj_repro_labels[::2] + traj_repro_labels[1:][::2]

    vis.error_boxplot(gt_repro_errs, gt_repro_labels, title='reprojection_error_static_ground_truth.png')
    vis.error_boxplot(traj_repro_errs, traj_repro_labels, title='reprojection_error_dynamic.png')
    vis.error_boxplot(match_repro_errs, match_repro_labels, title='reprojection_error_static.png')
    plt.show()

    # plot error histograms
    fig_gt_hist, gt_hist_axs = plt.subplots(2, 1, sharex=True, sharey=True)
    xlim = np.max(np.hstack(gt_repro_errs))
    for i, (cam_err1, cam_label1, cam_err2, cam_label2) in enumerate(zip(gt_err1, gt_label1, gt_err2, gt_label2)):
        vis.error_histogram(cam_err1, cam_err2, labels=[cam_label1, cam_label2],ax=gt_hist_axs[i],title='cam '+str(i),xlim=xlim)
    fig_gt_hist.suptitle('reprojection_error_histogram_static_ground_truth')
    fig_gt_hist.savefig(os.path.join(output_dir,'reprojection_error_histogram_static_ground_truth.png'))
    plt.show()

    fig_traj_hist, traj_hist_axs = plt.subplots(2, 1, sharex=True, sharey=True)
    xlim = np.max(np.hstack(traj_repro_errs))
    for i, (cam_err1, cam_label1, cam_err2, cam_label2) in enumerate(zip(traj_err1, traj_label1, traj_err2, traj_label2)):
        vis.error_histogram(cam_err1, cam_err2, labels=[cam_label1, cam_label2],ax=traj_hist_axs[i],title='cam '+str(i),xlim=xlim)
    fig_traj_hist.suptitle('reprojection_error_histogram_dynamic')
    fig_traj_hist.savefig(os.path.join(output_dir,'reprojection_error_histogram_dynamic.png'))
    plt.show()

    fig_match_hist, match_hist_axs = plt.subplots(2, 1, sharex=True, sharey=True)
    xlim = np.max(np.hstack(match_repro_errs))
    for i, (cam_err1, cam_label1, cam_err2, cam_label2) in enumerate(zip(match_err1, match_label1, match_err2, match_label2)):
        vis.error_histogram(cam_err1, cam_err2, labels=[cam_label1, cam_label2],ax=match_hist_axs[i],title='cam '+str(i),xlim=xlim)
    fig_match_hist.suptitle('reprojection_error_histogram_static')
    fig_match_hist.savefig(os.path.join(output_dir,'reprojection_error_histogram_static.png'))
    plt.show()


    # print('\n#################################################################\n')
    # print("Reconstructions from static-then-dynamic setting")
    # gt_dynamic4 = [gt_dynamic[x].copy() for x in flight_static_then_dynamic.sequence]
    # gt_dynamic4 = convert_timestamps(gt_dynamic4, flight_static_then_dynamic.alpha, flight_static_then_dynamic.beta)
    # reproject_ground_truth(flight_static_then_dynamic.cameras, gt_static, gt_dynamic4, output_dir=output_dir, prefix='static_then_dynamic_', flight=flight_static_then_dynamic)

    print('Finish!')

if __name__ == '__main__':
    main()