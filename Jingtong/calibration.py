import numpy as np
import cv2
import video
import scipy.io as scio


# Extract and save frames from a video for calibration
video_file = 'C:\\Users\\tong2\\Desktop\\calibration\\fixposition\\cam4\\Fx_Pt_C4.MP4'
img_folder = 'C:\\Users\\tong2\\Desktop\\calibration\\calibration_imgs\\'

frames = np.arange(1,1002,5)
imgs = video.getFrame(video_file,frames)

for i in range(len(imgs)):
    filename = 'img_{}.png'.format(frames[i])
    cv2.imwrite(img_folder+filename, imgs[i])


# Example of load calibration matrix from mat file
# camera = scio.loadmat('C:/Users/tong2/MyStudy/ETH/2019FS/Thesis/data/calibration/first_flight/phone_1/calibration.mat')
# print('\nIntrinsic parameters:\n',camera['intrinsic'])

print('\nFinished\n')
