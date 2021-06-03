# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import re
import numpy as np
import cv2
import argparse

'''
This script verifies drone detections of a given video by playing the video with the detections in background
'''

a = argparse.ArgumentParser()
a.add_argument("--detection_path", required=True, type=str, help="Path to the detections")
a.add_argument("--video_path", required=True, type=str, help="Path to the video")

args = a.parse_args()

# Read a video and its corresponding detection
# detection_path = ''
# video_path = ''
detection_path = args.detection_path
video_path = args.video_path

# Create a mask for detections in the same size of the video
detection = np.loadtxt(detection_path,usecols=(2,0,1)).T.astype(int)
cap = cv2.VideoCapture(video_path)
# cap.set(cv2.CAP_PROP_POS_FRAMES,100)
frame_0 = cap.read()[1]
traj = np.zeros_like(frame_0)
for i in range(detection.shape[1]):
    traj[detection[2,i],detection[1,i],0] = 1

# Plot all detections in each frame
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         # frame = cv2.flip(frame,0)
#         frame[traj[:,:,0]==1]=np.array([0,0,255])     # Color of the traj can be specified
#         frame = cv2.resize(frame,(1400,570))
#         cv2.imshow('Check Dectections, Press \'q\' to end',frame)

#     if cv2.waitKey(1)==ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # frame = cv2.flip(frame,0)
        frame[traj[:,:,0]==1]=np.array([0,0,255])     # Color of the traj can be specified
        dets = detection[:,(detection[0] == frame_id) | (detection[0] == frame_id + 500)]
        if dets.shape[1] > 0:
            for det in dets.T:
                if det[0] <= 500:
                    cv2.circle(frame, (int(det[1]), int(det[2])), radius=5, color=(0,0,255), thickness=-1)
                else:
                    cv2.circle(frame, (int(det[1]), int(det[2])), radius=5, color=(0,255,0), thickness=-1)
        frame = cv2.resize(frame,(1400,570))
        cv2.imshow('Check Dectections, Press \'q\' to end',frame)
        # print(frame_id)
        frame_id +=1

    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


print('Finish!')