import numpy as np
import cv2
from matplotlib import pyplot as plt
import video
import epipolar as ep
import visualization as vis


'''
This script tests the case of estimating F from two frames
'''

# Get path of videos
video1_path = 'C:/Users/tong2/MyStudy/ETH/2019FS/Thesis/data/C0028.MP4'
video2_path = 'C:/Users/tong2/MyStudy/ETH/2019FS/Thesis/data/VID_20190115_140427.MP4'

# Get images from videos, specify the frame number
img1 = video.getFrame(video1_path, 700)
img2 = video.getFrame(video2_path, 700)

# Convert images from RGB to graysclae
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Extract SIFT features
kp1, des1 = ep.extract_SIFT_feature(img1, [(0, 700), (1900, 930)])
kp2, des2 = ep.extract_SIFT_feature(img2, [(0, 700), (1900, 930)])

# Show results of SIFT feature extraction
print('\n\n{} features are extracted from image 1'.format(len(kp1)))
print('{} features are extracted from image 2'.format(len(kp2)))
img1_sift = cv2.drawKeypoints(img1,kp1,np.array([]))
img2_sift = cv2.drawKeypoints(img2,kp2,np.array([]))
cv2.namedWindow('SIFT in img1',cv2.WINDOW_NORMAL)
cv2.namedWindow('SIFT in img2',cv2.WINDOW_NORMAL)
cv2.imshow('SIFT in img1',img1_sift)
cv2.imshow('SIFT in img2',img2_sift)
cv2.waitKey()


# Match features
pts1, pts2, matches, goodmatch = ep.matching_feature(kp1, kp2, des1, des2, ratio=0.95)

# Show results of feature matching
print('\n{} features are matched'.format(sum(np.array(goodmatch)[:,0])))
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = goodmatch,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

img_match = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
cv2.namedWindow('Matches',cv2.WINDOW_NORMAL)
cv2.imshow('Matches',img_match)
cv2.waitKey()


# Compute fundametal matrix F and return inlier matches(optional)
F, mask = ep.computeFundamentalMat(pts1, pts2, error=10)
pts1 = np.int32(pts1)[mask.ravel()==1]
pts2 = np.int32(pts2)[mask.ravel()==1]

print('\n{} feature correspondences are valid after estimating F'.format(pts1.shape[0]))

# Draw epipolar lines
vis.plotEpiline(img1, img2, pts1, pts2, F)
cv2.destroyAllWindows()
