import numpy as np
import cv2
# from matplotlib import pyplot as plt
import epipolar as ep

'''
This script tests the case of using multiple frames to estimate F
'''

# Parameters
match_ratio = 0.9
RansacReproErr = 10
numFrame = 5
rangeFrame = 1000

# Input Videos
video1_path = 'data/video_1.mp4'
video2_path = 'data/video_2.mp4'

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams,searchParams)


Frames = np.random.randint(0,rangeFrame,numFrame)
P1 = np.array([])
P2 = np.array([])
for i in range(numFrame):
    cap1.set(cv2.CAP_PROP_POS_FRAMES,Frames[i])
    cap2.set(cv2.CAP_PROP_POS_FRAMES,Frames[i])

    read1,img1 = cap1.read()
    read2,img2 = cap2.read()
    
    # Convert images from RGB to graysclae
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Extract SIFT features
    kp1, des1 = ep.extract_SIFT_feature(img1, [(0, 600), (1900, 930)])
    kp2, des2 = ep.extract_SIFT_feature(img2, [(0, 600), (1900, 930)])

    # Match features
    pts1, pts2, matches, goodmatch = ep.matching_feature(kp1, kp2, des1, des2, ratio=match_ratio)

    # Show result of current frame
    print('\nProcessing {} of {} frames: the frame {}'.format(i+1,numFrame,Frames[i]))
    print('{} features are matched'.format(sum(np.array(goodmatch)[:,0])))

    # Accumulate features
    if i==0:
        P1 = pts1
        P2 = pts2
    else:
        P1 = np.vstack((P1,pts1))
        P2 = np.vstack((P2,pts2))

# Compute fundametal matrix F and return inlier matches(optional)
print('\n\n{} feature correspondences are valid Before estimating F'.format(P1.shape[0]))
F, mask = ep.computeFundamentalMat(P1, P2, error=RansacReproErr)
P1 = np.int32(P1)[mask.ravel()==1]
P2 = np.int32(P2)[mask.ravel()==1]
print('{} feature correspondences are valid After estimating F'.format(P1.shape[0]))

# Draw epipolar lines
ep.plotEpiline(img1, img2, P1, P2, F)

cap1.release()
cap2.release()
cv2.destroyAllWindows()