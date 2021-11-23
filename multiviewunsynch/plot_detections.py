import numpy as np
import pickle
from datetime import datetime
from reconstruction import synchronization as sync

from tools.util import match_overlap, homogeneous
from matplotlib import pyplot as plt
import os
import cv2
import argparse
from glob import glob
import json

def plot_detections(img, raw_dets, used_dets, figname):
    # plt.figure(figsize=(12, 10))
    plt.figure()
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        h,w,_ = img.shape
        plt.xlim([0,w])
        plt.ylim([h,0])
    
    # plot raw detections
    plt.scatter(raw_dets[0], raw_dets[1], c='yellow', marker='o', s=50)
    plt.scatter(used_dets[0], used_dets[1], c='orange', marker='o', s=50)

    plt.axis('off')
    plt.savefig(figname,bbox_inches='tight',pad_inches = 0)

def main():
    # detection path
    det_path = ['/scratch2/wuti/Repos/mvus/experiments/webcam-datasets/nyc_set17/det_opencv/cam0_set17.txt', '/scratch2/wuti/Repos/mvus/experiments/webcam-datasets/nyc_set17/det_opencv/cam2_set17.txt']
    
    # output dir
    output_dir = '/scratch2/wuti/Repos/mvus/experiments/webcam-datasets/nyc_set17/teaser'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # result path
    result_path = '/scratch2/wuti/Repos/mvus/experiments/webcam-datasets/nyc_set17/static_dynamic/nyc_static_dynamic_superglue_10.pkl'
    
    # load result
    with open(result_path, 'rb') as file:
        flight = pickle.load(file)

    raw_dets = [flight.detections[x] for x in flight.sequence]
    flight.settings['undist_points'] = False
    flight.detection_to_global()
    sync_dets = [flight.detections_global[x] for x in flight.sequence]

    timestamp = raw_dets[flight.ref_cam][0]
    used_dets = []
    
    for sync_det in sync_dets:
        # sample to the ref camera
        _, used_det = match_overlap(raw_dets[flight.ref_cam], sync_det)
        used_dets.append(used_det)
        timestamp = np.intersect1d(timestamp, used_det[0])
    
    used_dets = list(map(lambda x: x[1:,np.isin(x[0], timestamp)], used_dets))

    for i, (cam, raw_det, used_det) in enumerate(zip(flight.cameras, raw_dets, used_dets)):
        print(f'Plot the trajectory of camera {i}...')
        # print(used_det.shape)
        plot_detections(cam.img, raw_det[1:], used_det, os.path.join(output_dir, f'traj_cam{i}.png'))


if __name__ == '__main__':
    main()