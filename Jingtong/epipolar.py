import numpy as np
import cv2
from matplotlib import pyplot as plt


def extract_SIFT_feature(img, mask_range=None):
    '''
    Function:
            extract SIFT features from input image
    Input:
            img = input image
            mask_range = a list with length 2, which describes the region of interest of img,
                         containing coordinates of the top-left and the down-right points,
                         None by default
    Output:
            kp = list of keypoints
            des = list of descriptors
    '''

    sift = cv2.xfeatures2d.SIFT_create()

    if mask_range == None:
        mask = None
    else:
        mask = np.zeros(img.shape, dtype=img.dtype)
        cv2.rectangle(mask, mask_range[0], mask_range[1], (255), thickness = -1)

    kp, des = sift.detectAndCompute(img, mask)
    return kp, des


def matching_feature(kp1, kp2, des1, des2, method=1, ratio=0.7):
    '''
    Function:
            matching features that are extracted in two images
    Input:
            kp1,kp2,des1,des2: = keypoints and their descriptors in two images
            method = 1: FLANN Matcher (default)
                     0: Bruto-Force Matcher
            ratio = threshold for ratio of similartiy measure between the best match
                    and the second best match, only for FLANN Matcher, 0.7 by default
    Output:
            pts1 = pixel coordinates of corresponding features in img1
            pts2 = pixel coordinates of corresponding features in img2,
                   which has the same size as pts1
            matches = the Matcher object
            matchesMask = index of good matches, only for FLANN Matcher
    '''

    pts1 = []
    pts2 = []

    if method:
        # FLANN Matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # record good matches
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < ratio*n.distance:
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
                matchesMask[i]=[1,0]
        
        return pts1, pts2, matches, matchesMask

    else:
        # Brute Force Matching
        bf = cv2.BFMatcher(crossCheck=True)
        matches = bf.match(des1,des2)

        # Use every matches
        for m in matches:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
        return pts1, pts2, matches


def computeFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, error=3, inliers=True):
    '''
    Function:
            compute fundamental matrix given correspondences (at least 8)
    Input:
            pts1, pts2 = list of pixel coordinates of corresponding features
            method = cv2.FM_RANSAC: Using RANSAC algorithm (default)
                     cv2.FM_LMEDS: Using least-median algorithm
                     cv2.FM_8POINT: Using 8 points algorithm
            error = reprojection threshold that describes maximal distance from a 
                    point to a epipolar line
            inlier = True: return F and the mask for inliers
                     False: only reture F
    Output:
            F = Fundamental matrix with size 3*3
            mask = index for inlier correspondences (optional)
    '''

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(pts1,pts2,method,error)

    if inliers:
        return F, mask
    else:
        return F


def normalize_2d_points(x):
    '''
    Function:
            normalize input points such that mean = 0 and distance to center = sqrt(2)
    Input:
            x = 2D points in numpy array
    Output:
            x_n = normalized 2D points in form of 3*N
            T = 3x3 normalization matrix
                (s.t. x_n=T*x when x is in homogenous coords)
    '''

    # Make sure x has the form of 3*N
    if x.shape[0]==2:
        x = np.vstack((x,np.ones(x.shape[1])))
    elif x.shape[1] == 2:
        x = np.hstack((x,np.ones(x.shape[0]).reshape(-1,1))).T
    elif x.shape[1] == 3:
        x = x.T
    
    # Calculate mean and scale
    x_mean = np.mean(x[:2],axis=1)
    x_scale = np.sqrt(2) / np.std(x[:2])

    # Create normalization matrix T
    T = np.array([[x_scale,0,-x_scale*x_mean[0]],[0,x_scale,-x_scale*x_mean[1]],[0,0,1]])
    x_n = np.dot(T,x)

    return x_n, T


def compute_fundamental(x1,x2):
    '''
    Compute fundamental matrix from 2d points in image coordinates.

    Input points do not need to be normalized in advance.
    '''

    # Check that x1,x2 have same number of points
    num = x1.shape[1]
    if x2.shape[1] != num:
        raise ValueError("Number of points do not match!")
    elif num < 8:
        raise ValueError("At least 8 points needed!")

    # Normalize input points
    x1, T1 = normalize_2d_points(x1)
    x2, T2 = normalize_2d_points(x2)

    # Design matrix A
    A = np.array([x1[0]*x2[0],x1[0]*x2[1],x1[0],
                  x1[1]*x2[0],x1[1]*x2[1],x1[1],
                  x2[0],x2[1],np.ones(x1.shape[1])]).T
    
    # Solve F by SVD
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)

    # Constrain of det(F)=0
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(np.dot(U,np.diag(S)),V)

    # Denormalization
    F = np.dot(np.dot(T1.T,F),T2)

    return F.T/F[2,2]


def compute_essential(x1,x2):
    '''
    Compute essential matrix from 2d points correspondences, 
    
    which have to be normalized by calibration matrix K in advance.
    '''

    # Check that x1,x2 have same number of points
    num = x1.shape[1]
    if x2.shape[1] != num:
        raise ValueError("Number of points do not match!")
    elif num < 8:
        raise ValueError("At least 8 points needed!")

    # Normalize input points
    x1, T1 = normalize_2d_points(x1)
    x2, T2 = normalize_2d_points(x2)

    # Design matrix A
    A = np.array([x1[0]*x2[0],x1[0]*x2[1],x1[0],
                  x1[1]*x2[0],x1[1]*x2[1],x1[1],
                  x2[0],x2[1],np.ones(x1.shape[1])]).T
    
    # Solve F by SVD
    U,S,V = np.linalg.svd(A)
    E = V[-1].reshape(3,3)

    # Constrain of det(E)=0 and first two singular values are equal (set to 1)
    U,S,V = np.linalg.svd(E)
    S[0], S[1], S[2] = 1, 1, 0
    E = np.dot(np.dot(U,np.diag(S)),V)

    # Denormalization
    E = np.dot(np.dot(T1.T,E),T2)

    return E.T/E[2,2]


def Sampson_error(x1,x2,F):
    Fx1 = np.dot(F,x1)
    Fx2 = np.dot(F,x2)

    w = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
    error = np.diag(np.dot(np.dot(x2.T, F),x1))**2 / w

    return error


def compute_P_from_F(F):
    # Compute the left epipole
    U,S,V = np.linalg.svd(F.T)
    e = V[-1]
    e = e/e[2]

    Te = np.array([[0,-e[2],e[1]],[e[2],0,-e[0]],[-e[1],e[0],0]])
    P = np.vstack((np.dot(Te,F.T).T,e)).T
    return P


def drawlines(img1,img2,lines,pts1,pts2):
    ''' 
    Function:
            draw lines and circles on image pair, specifically for epipolar lines
    Input:
            img1 = image on which we draw (epipolar) lines for the points in img2
            img2 = corresponding image
            lines = corresponding epipolar lines 
            pts1,pts2 = corresponding feature points
    Output:
            img1 = image1 with lines and circles
            img2 = image2 with circles 
    '''
    r,c = img1.shape

    # Convert grayscale to BGR if needed
    if len(img1.shape)==2:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    if len(img2.shape)==2:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    # Draw epipolar lines in image1 and corresponding feature points in both images
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def plotEpiline(img1, img2, pts1, pts2, F):
    '''
    Function:
            plot epipolar lines in both images
    Input:
            img1, img2 = image pair
            pts1, pts2 = corresponding feature points
            F = fundamental matrix
    Output:
            plot epipolar lines on image pair
    '''
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img3 = drawlines(img1,img2,lines1,pts1,pts2)[0]

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img5 = drawlines(img2,img1,lines2,pts2,pts1)[0]

    # Show results
    cv2.namedWindow('Epipolar lines in img1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Epipolar lines in img1',500,300)
    cv2.imshow('Epipolar lines in img1',img3)
    cv2.namedWindow('Epipolar lines in img2',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Epipolar lines in img2',500,300)
    cv2.imshow('Epipolar lines in img2',img5)
    cv2.waitKey(0)


if __name__ == "__main__":

    ''' Test '''
    img1 = cv2.imread('C:/Users/tong2/MyStudy/ETH/2018HS/ComputerVision/lab/lab04/cv_lab04_model_fitting/src/epipolar_geometry/images/pumpkin1.jpg',0)
    img2 = cv2.imread('C:/Users/tong2/MyStudy/ETH/2018HS/ComputerVision/lab/lab04/cv_lab04_model_fitting/src/epipolar_geometry/images/pumpkin2.jpg',0)

    # Extract SIFT features
    kp1, des1 = extract_SIFT_feature(img1)
    kp2, des2 = extract_SIFT_feature(img2)

    # Match features
    pts1, pts2, matches, matchesMask = matching_feature(kp1, kp2, des1, des2, ratio=0.8)

    # Compute fundametal matrix F and return inlier mask(optional)
    F, mask = computeFundamentalMat(pts1, pts2)

    pts1 = np.int32(pts1)[mask.ravel()==1]
    pts2 = np.int32(pts2)[mask.ravel()==1]

    # Draw epipolar lines
    plotEpiline(img1, img2, pts1, pts2, F)
