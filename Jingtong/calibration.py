import numpy as np
import cv2
import video
import scipy.io as scio


# Extract and save frames from a video for calibration
video_file = 'C:\\Users\\tong2\\Desktop\\calibration\\VID_20190429_141058.mp4'
img_folder = 'C:\\Users\\tong2\\Desktop\\calibration\\'

frames = np.arange(0,1001,100)
imgs = video.getFrame(video_file,frames)

for i in range(len(imgs)):
    filename = 'img_{}.png'.format(frames[i])
    cv2.imwrite(img_folder+filename, imgs[i])


# Example of load calibration matrix from mat file
# camera_1 = scio.loadmat('C:/Users/tong2/MyStudy/ETH/2019FS/Thesis/data/calibration/first_flight/phone_1/intrinsic.mat')
# print('\nIntrinsic parameters:\n',camera_1['intrinsic'])

print('\nFinished\n')
