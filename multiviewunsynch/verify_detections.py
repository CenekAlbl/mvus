import numpy as np
import cv2

'''
This script verifies drone detections of a given video by playing the video with the detections in background
'''

# Read a video and its corresponding detection
detection_path = '../data/paper/crash/detection/out_xiaomi_1_30.txt'
video_path = 'C:/Users/tong2/MyStudy/ETH/2019FS/Thesis/ICRA/crash/Xiaomi_mi9.mp4'

# Create a mask for detections in the same size of the video
detection = np.loadtxt(detection_path[1:],usecols=(2,0,1)).T.astype(int)
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES,100)
frame_0 = cap.read()[1]
traj = np.zeros_like(frame_0)
for i in range(detection.shape[1]):
    traj[detection[2,i],detection[1,i],0] = 1

# Plot all detections in each frame
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # frame = cv2.flip(frame,0)
        frame[traj[:,:,0]==1]=np.array([255,0,0])     # Color of the traj can be specified
        frame = cv2.resize(frame,(1400,570))
        cv2.imshow('Check Dectections, Press \'q\' to end',frame)

    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


print('Finish!')