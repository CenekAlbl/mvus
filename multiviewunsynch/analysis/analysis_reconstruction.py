import numpy as np
import pickle
import tools.visualization as vis
from datetime import datetime
from analysis.compare_gt import align_gt
from reconstruction import synchronization as sync

from reconstruction import epipolar as ep

from itertools import combinations
from matplotlib import pyplot as plt

def reproject_ground_truth(cameras, gt_pts ,n_bins=10, prefix=''):
    '''
    Function:
        Compute and plot the reprojection errors of the ground truth matches
    Input:
        cameras = the list of Camera objects to be evaluated
        gt_pts = the ground truth matches
        n_bins = number of bins for plotting the histogram of the errors
    '''
    
    # split gt_pts into two parts
    gt_pts_parts = np.split(gt_pts, len(cameras))

    combos = combinations(range(len(cameras)),2)

    repo_errors = []
    # for every pair of cameras, triangulate 3d points and reproject
    for t1, t2 in combos:
        print("triangulate camera pairs: (%d, %d)" %(t1, t2))
        # undistort gt_points
        gt_un1 = cameras[t1].undist_point(gt_pts_parts[t1].T)
        gt_un2 = cameras[t2].undist_point(gt_pts_parts[t2].T)

        # triangulate
        X_gt = ep.triangulate_matlab(gt_un1, gt_un2, cameras[t1].P, cameras[t2].P)

        # backproject the triangulated points on the images
        gt_repo1 = cameras[t1].dist_point3d(X_gt[:-1].T)
        gt_repo2 = cameras[t2].dist_point3d(X_gt[:-1].T)
        # compute the error
        repo_err1 = ep.reprojection_error(gt_un1, cameras[t1].projectPoint(X_gt))
        repo_err2 = ep.reprojection_error(gt_un2, cameras[t2].projectPoint(X_gt))
        repo_errors.append([repo_err1, repo_err2])
        print("mean reprojection error of camera pair (%d, %d) is %f and %f" %(t1, t2, np.mean(repo_err1), np.mean(repo_err2)))

        # histogram of reprojection error
        fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)
        axs[0].hist(repo_err1, bins=n_bins)
        axs[1].hist(repo_err2, bins=n_bins)
        plt.savefig(prefix+'repo_cam{}_{}.png'.format(t1, t2))

        # plot images
        if prefix == 'dynamic_only_':
            vis.show_2D_all(gt_pts_parts[t1].T, gt_repo1, title=prefix+'cam'+str(t1)+' ground truth reprojection', color=True, line=False)
            vis.show_2D_all(gt_pts_parts[t2].T, gt_repo2, title=prefix+'cam'+str(t2)+' ground truth reprojection', color=True, line=False)
        else:
            vis.show_2D_all(gt_pts_parts[t1].T, gt_repo1, title=prefix+'cam'+str(t1)+' ground truth reprojection', color=True, line=False, bg=cameras[t1].img)
            vis.show_2D_all(gt_pts_parts[t2].T, gt_repo2, title=prefix+'cam'+str(t2)+' ground truth reprojection', color=True, line=False, bg=cameras[t2].img)

def main():
    # Load ground truth
    gt_static_file = '/scratch2/wuti/Repos/3D-Object-Trajectory-Reconstruction-Webcam/multiviewunsynch/webcam-datasets/nyc-datasets/static_gt/static.txt'
    # gt_dynamic_file = ''

    gt_static = np.loadtxt(gt_static_file, delimiter=' ')
    # gt_dynamic = np.loadtxt(gt_dynamic_file, delimiter=' ')

    # Load scenes
    data_file_dynamic = '/scratch2/wuti/Repos/mvus/experiments/dynamic/nyc_colmap_dynamic_ori_30.pkl'
    data_file_static = '/scratch2/wuti/Repos/mvus/experiments/static/nyc_colmap_static_superglue_30.pkl'
    data_file_static_dynamic = '/scratch2/wuti/Repos/mvus/experiments/static_dynamic/nyc_colmap_static_dynamic_superglue_30.pkl'

    with open(data_file_dynamic, 'rb') as file:
        flight_dynamic = pickle.load(file)

    with open(data_file_static, 'rb') as file:
        flight_static = pickle.load(file)

    with open(data_file_static_dynamic, 'rb') as file:
        flight_static_dynamic = pickle.load(file)

    # Analysis
    # 2D reprojection error
    print("Plot reprojection error")
    # dynamic only
    print("Reconstructions from dynamic-only setting")
    reproject_ground_truth(flight_dynamic.cameras, gt_static, prefix='dynamic_only_')
    print('\n#################################################################\n')
    print("Reconstructions from static-only setting")
    reproject_ground_truth(flight_static.cameras, gt_static, prefix='static_only_')
    print('\n#################################################################\n')
    print("Reconstructions from static-dynamic setting")
    reproject_ground_truth(flight_static_dynamic.cameras, gt_static, prefix='static_dynamic_')

    print('Finish!')

if __name__ == '__main__':
    main()