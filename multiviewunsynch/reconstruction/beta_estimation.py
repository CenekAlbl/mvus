# This script estimates the time shift (beta) between two seires of detections.
# It is assumed that both series have the same fps and a rough time shift is given.

import numpy as np
from datetime import datetime
import cv2
from scipy.interpolate import splprep,splev


def find_intervals(x):
    '''
    Given indices of detections, return a matrix that contains the start and the end of each
    continues part
    '''

    assert len(x.shape)==1, 'Input must be a 1D-array'

    interval = np.array([[x[0]],[np.nan]])
    for i in range(1,len(x)):
        if x[i]-x[i-1] != 1:
            interval[1,-1] = x[i-1]
            interval = np.append(interval,[[x[i]],[np.nan]],axis=1)
    interval[1,-1] = x[-1]
    return interval


def filter_detections(idx,interval):
    '''
    Filter out indices that do not belong to any given intervals
    '''

    assert len(idx.shape)==1, 'Detecion indices must be a 1D-array'
    assert interval.shape[0]==2, 'Intervals must be a 2D-array'

    idx_filter = np.array([])
    for i in idx:
        for j in range(interval.shape[1]):
            if i >= interval[0,j] and i <= interval[1,j]:
                idx_filter = np.append(idx_filter,i)
                break
    return idx_filter


def search_beta(detect_1,detect_2,beta_prior,search=50,step=1,error=8):
    start = datetime.now()

    beta_list = np.arange(beta_prior-search, beta_prior+search,step)

    tck_1, u_1 = splprep(detect_1[1:],u=detect_1[0],s=0,k=3)
    tck_2, u_2 = splprep(detect_2[1:],u=detect_2[0],s=0,k=3)

    int_2 = find_intervals(u_2)
    
    inlier_ratio_max = 0
    beta_final = np.nan
    for beta in beta_list:
        idx_2 = filter_detections(u_1+beta, int_2)
        idx_1 = idx_2-beta

        num_pts = len(idx_1)
        if len(idx_1) < 30*5:
            continue

        pts_1 = np.asarray(splev(idx_1,tck_1))
        pts_2 = np.asarray(splev(idx_2,tck_2))

        F, mask = cv2.findFundamentalMat(pts_1.T, pts_2.T, method=cv2.FM_RANSAC, ransacReprojThreshold=error)
        inlier_ratio = sum(mask.reshape(-1,)) / num_pts

        if inlier_ratio > inlier_ratio_max:
            inlier_ratio_max = inlier_ratio
            beta_final = beta

    print('Ratio of inlier: {}'.format(inlier_ratio_max))
    print('\nTime: {}\n'.format(datetime.now()-start))

    return beta_final, inlier_ratio_max, num_pts


def estimate_beta(detect_1,detect_2,beta_prior=0):

    if not beta_prior:
        t = int(max(abs(detect_1[0,0]-detect_2[0,-1]),abs(detect_1[0,-1]-detect_2[0,0])) / 2)
    else:
        t = 150

    beta_est, ratio, num = search_beta(detect_1,detect_2,beta_prior,search=t,step=30)
    beta_est, ratio, num = search_beta(detect_1,detect_2,beta_est,search=15,step=1)
    beta_est, ratio, num = search_beta(detect_1,detect_2,beta_est,search=0.5,step=0.01)

    return beta_est, ratio, num



if __name__ == "__main__":


    '''
    Ref:    mate10

    Tar:    sonyg1 = -467.00
            sonyalpha5001 = 409.49
            mate7 = 458.40
            gopro1 = -552.17
            sony5n1 = -251.00
    '''

    path_detect_1 = './data/paper/fixposition/detection/outp_mate10_1_30.txt'
    path_detect_2 = './data/paper/fixposition/detection/outp_sony5n1_30.txt'
    beta_prior = -250
    
    detect_1 = np.loadtxt(path_detect_1,usecols=(2,0,1)).T
    detect_2 = np.loadtxt(path_detect_2,usecols=(2,0,1)).T

    beta_est, ratio, num = estimate_beta(detect_1,detect_2,beta_prior=beta_prior)

    # tck_1, u_1 = splprep(detect_1[1:],u=detect_1[0],s=0,k=3)
    # tck_2, u_2 = splprep(detect_2[1:],u=detect_2[0],s=0,k=3)

    # int_2 = find_intervals(u_2)
    # idx_2 = filter_detections(u_1+beta_est,int_2)
    # idx_1 = idx_2-beta_est

    # pts_1 = np.asarray(splev(idx_1,tck_1))
    # pts_2 = np.asarray(splev(idx_2,tck_2))
    
    # F, mask = cv2.findFundamentalMat(pts_1.T, pts_2.T, method=cv2.FM_RANSAC, ransacReprojThreshold=5)
    # num_pts = pts_1.shape[1]
    # inlier_ratio = sum(mask.reshape(-1,)) / num_pts




    print('Finish !')