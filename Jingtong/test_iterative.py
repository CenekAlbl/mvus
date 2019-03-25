import numpy as np
import cv2
import epipolar as ep
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
This script tests the case of iteratively using image pairs to estimate F
'''


'''
Parameters
'''
match_ratio = 0.8
RansacReproErr = 10
Frame_interval = 30

Use_traj = True
start_traj_1 = 153
start_traj_2 = 71
num_traj = 1500

calibration = open('data/results.pickle','rb')
K1 = np.asarray(pickle.load(calibration)["intrinsic_matrix"])
K2 = K1

'''
Main implementation
'''

# Input Videos
video1_path = 'data/video_1.mp4'
video2_path = 'data/video_2.mp4'

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# Create SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# Create FLANN matcher
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams,searchParams)

# Load drone trajectories
if Use_traj:
    traj_1 = np.loadtxt('data/video_1_output.txt',skiprows=1,dtype=np.int32)
    traj_2 = np.loadtxt('data/video_2_output.txt',skiprows=1,dtype=np.int32)

    traj = np.hstack((traj_1[start_traj_1:start_traj_1+num_traj,1:],traj_2[start_traj_2:start_traj_2+num_traj,1:]))
    traj = np.unique(traj,axis=0)

# Starting
time = 1
while time:
    cap1.set(cv2.CAP_PROP_POS_FRAMES,time)
    cap2.set(cv2.CAP_PROP_POS_FRAMES,time)
    print('\n\n-------------Frame:{}-------------'.format(time))

    read1,img1 = cap1.read()
    read2,img2 = cap2.read()
    
    # Convert images from RGB to graysclae
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Extract SIFT features
    kp1, des1 = ep.extract_SIFT_feature(img1, [(0, 600), (1900, 930)])
    kp2, des2 = ep.extract_SIFT_feature(img2, [(0, 600), (1900, 930)])

    # Show results of SIFT feature extraction
    print('\n{} features are extracted from image 1'.format(len(kp1)))
    print('{} features are extracted from image 2'.format(len(kp2)))
    img1_sift = cv2.drawKeypoints(img1,kp1,np.array([]))
    img2_sift = cv2.drawKeypoints(img2,kp2,np.array([]))
    cv2.namedWindow('SIFT in img1',cv2.WINDOW_NORMAL)
    cv2.namedWindow('SIFT in img2',cv2.WINDOW_NORMAL)
    cv2.imshow('SIFT in img1',img1_sift)
    cv2.imshow('SIFT in img2',img2_sift)
    cv2.waitKey()
    
    # Match features
    print("\nMatching features...")
    pts1, pts2, matches, goodmatch = ep.matching_feature(kp1, kp2, des1, des2, ratio=match_ratio)

    # Show results of feature matching
    print('{} features are matched'.format(sum(np.array(goodmatch)[:,0])))
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = goodmatch,
                    flags = cv2.DrawMatchesFlags_DEFAULT)

    img_match = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    cv2.namedWindow('Matches',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Matches',800,300)
    cv2.imshow('Matches',img_match)
    cv2.waitKey()

    # Add trajectory matches
    if Use_traj:
        print('\n{} matches from drone trajectory are appended'.format(traj.shape[0]))
        if pts1:
            pts1 = np.vstack((np.int32(pts1),traj[:,:2]))
            pts2 = np.vstack((np.int32(pts2),traj[:,2:]))
        else:
            pts1 = traj[:,:2]
            pts2 = traj[:,2:]

    # Compute fundametal matrix F and return inlier matches(optional)
    F, mask = ep.computeFundamentalMat(pts1, pts2, error=RansacReproErr)
    pts1 = np.int32(pts1)[mask.ravel()==1]
    pts2 = np.int32(pts2)[mask.ravel()==1]
    print('\n{} feature correspondences are valid after estimating F'.format(pts1.shape[0]))

    # Draw epipolar lines
    ep.plotEpiline(img1, img2, pts1, pts2, F)

    time = time + Frame_interval

    # Use ESC to exit
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break

cap1.release()
cap2.release()


# Compute essential matrix E from fundamental matrix F
E = np.dot(np.dot(K2.T,F),K1)
num, R, t, mask = cv2.recoverPose(E,pts1,pts2,K1)
P1 = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
P2 = np.dot(K2,np.hstack((R,t)))

# Compute projection matrix P from fundamental matrix F
# P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
# P2 = ep.compute_P_from_F(F)

# Triangulate points
print("\nTriangulating feature points...")
pts1=pts1.astype(np.float64)
pts2=pts2.astype(np.float64)
X = cv2.triangulatePoints(P1,P2,pts1.T,pts2.T)
X/=X[-1]

# Show 3D results
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[0],X[1],X[2])
plt.show()

cv2.destroyAllWindows()