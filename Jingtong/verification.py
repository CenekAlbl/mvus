import pickle
import cv2
import numpy as np
import epipolar as ep
import visualization as vis
# import ransac1
from matplotlib import pyplot as plt


def verify_fundamental(x1,x2,img1,img2,part=0.5,add_noise=True,noise_std=1,Use_Ransac=True,inlier_ratio=0.6):
    # Split data
    num_all = x1.shape[1]
    num_train = int(part*num_all)
    num_val = num_all - num_train

    idx = np.random.permutation(num_all)
    x1_train = x1[:,idx[:num_train]]
    x1_val = x1[:,idx[num_train:]]
    x2_train = x2[:,idx[:num_train]]
    x2_val = x2[:,idx[num_train:]]
    print("\n{} points are used to compute F, {} points to validate using Sampson distance".format(num_train,num_val))

    # Add noise (optional)
    if add_noise:
        noise_1 = np.random.randn(2, num_train) * noise_std
        noise_2 = np.random.randn(2, num_train) * noise_std
        inlier_should = int(num_train*inlier_ratio)
        noise_free = np.random.choice(num_train, inlier_should, replace=False)
        noise_1[:,noise_free] = 0
        noise_2[:,noise_free] = 0

        x1_train[:2] = x1_train[:2] + noise_1
        x2_train[:2] = x2_train[:2] + noise_2

        # Make sure that points are inside of images
        x1_train[0][x1_train[0]>img1.shape[1]] = img1.shape[1]
        x1_train[1][x1_train[1]>img1.shape[0]] = img1.shape[0]
        x1_train[0][x1_train[0]<0] = 0
        x1_train[1][x1_train[1]<0] = 0
        x2_train[0][x2_train[0]>img2.shape[1]] = img2.shape[1]
        x2_train[1][x2_train[1]>img2.shape[0]] = img2.shape[0]
        x2_train[0][x2_train[0]<0] = 0
        x2_train[1][x2_train[1]<0] = 0

    # Using Ransac (optional)
    if Use_Ransac:
        # Compute F using 8-points algorithm + Ransac
        F1, mask = cv2.findFundamentalMat(x1_train[:2].T, x2_train[:2].T, method=cv2.FM_RANSAC,ransacReprojThreshold=2,confidence=0.99)

        # inlier_should = int(num_train*inlier_ratio)
        # model = ransac1.Ransac_Fundamental()
        # F2, inliers = ransac1.F_from_Ransac(x1_train, x2_train, model, threshold=1e-2, inliers=int((inlier_should-8)*0.3))

        estimate = ep.compute_fundamental_Ransac(x1,x2,threshold=2,maxiter=1000,loRansac=True)
        F2 = estimate['model'].reshape((3,3))
        inliers = estimate['inliers']
    else:
        # Compute F using 8-points algorithm
        F1,mask = cv2.findFundamentalMat(x1_train[:2].T, x2_train[:2].T, method=cv2.FM_8POINT)
        F2 = ep.compute_fundamental(x1_train,x2_train)

    # Compute errors and compare
    error_1 = ep.Sampson_error(x1_val, x2_val, F1)
    error_2 = ep.Sampson_error(x1_val, x2_val, F2)

    print("\nThe mean residual from OpenCV is {}".format(np.mean(error_1)))
    print("\nThe mean residual from own implementation is {}".format(np.mean(error_2)))

    print('\nRatio of inliers from OpenCV is {}'.format(sum(mask.T[0]==1) / num_train))
    print('\nRatio of inliers from own implementation is {}'.format(len(inliers) / num_train))

    # Visualize epipolar lines
    vis.plot_epipolar_line(img1,img2,F1,x1_val,x2_val)
    vis.plot_epipolar_line(img1,img2,F2,x1_val,x2_val)


if __name__ == "__main__":
    '''
    Load data
    '''
    # Load trajectory data
    X = np.loadtxt('data/Real_Trajectory.txt')
    X_homo = np.insert(X,3,1,axis=0)

    # Load camera parameter
    with open('data/Synthetic_Camera.pickle', 'rb') as file:
        Camera =pickle.load(file)

    # Camara 1
    P1 = np.dot(Camera["K1"],np.hstack((Camera["R1"],Camera["t1"])))
    x1 = np.dot(P1,X_homo)
    x1 /= x1[-1]
    img1 = Camera["img1"]

    # Camara 2
    P2 = np.dot(Camera["K2"],np.hstack((Camera["R2"],Camera["t2"])))
    x2 = np.dot(P2,X_homo)
    x2 /= x2[-1]
    img2 = Camera["img2"]

    '''
    Verify the algorithm of computing fundamental matrix F with Ransac
    '''
    start_1 = 153
    start_2 = 71
    num_traj = 1500

    traj_1 = np.loadtxt('data/video_1_output.txt',skiprows=1,dtype=np.int32)
    traj_2 = np.loadtxt('data/video_2_output.txt',skiprows=1,dtype=np.int32)

    x1 = np.vstack((traj_1[start_1:start_1+num_traj,1:].T, np.ones(num_traj)))
    x2 = np.vstack((traj_2[start_2:start_2+num_traj,1:].T, np.ones(num_traj)))

    r,c = 1080,1920
    x1_int, x2_int = np.int16(x1), np.int16(x2)
    img1 = np.zeros((r,c),dtype=np.uint8)
    img2 = np.zeros((r,c),dtype=np.uint8)
    for i in range(num_traj):
        img1[x1_int[1,i],x1_int[0,i]]= 255
        img2[x2_int[1,i],x2_int[0,i]]= 255

    verify_fundamental(x1,x2,img1,img2,add_noise=False,part=0.8)