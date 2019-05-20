import numpy as np
import math
import cv2
import ransac
import visualization as vis
from scipy.optimize import least_squares, root


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

    pts1 = pts1[:2].T
    pts2 = pts2[:2].T

    F, mask = cv2.findFundamentalMat(pts1,pts2,method,error)

    if inliers:
        return F, mask.reshape(-1,)
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


def compute_fundamental_Ransac(x1,x2,threshold=10e-4,maxiter=500,verbose=False,loRansac=False):
    
    def model_function(data,param=None):
        s1 = data[:3]
        s2 = data[3:]
        F = compute_fundamental(s1,s2)
        return np.ravel(F)

    def error_function(M,data,param=None):
        s1 = data[:3]
        s2 = data[3:]
        F = M.reshape((3,3))
        return Sampson_error(s1,s2,F)

    data = np.append(x1,x2,axis=0)
    if loRansac:
        return ransac.loRansacSimple(model_function,error_function,data,8,threshold,maxiter,verbose=verbose)
    else:
        return ransac.vanillaRansac(model_function,error_function,data,8,threshold,maxiter,verbose=verbose)


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
    Fx2 = np.dot(F.T,x2)

    w = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
    error = np.diag(np.dot(np.dot(x2.T, F),x1))**2 / w

    return error


def skew(a):
    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])


def compute_epipole_from_F(F,left=False):
    '''
    Compute the epipole given the fundamental matrix, by default return the right epipole
    '''

    if left:
        F = F.T
    
    U,S,V = np.linalg.svd(F)
    e = V[-1]
    return e/e[2]


def compute_P_from_F(F):
    '''
    Compute P2 from the fundamental matrix, assuming P1 = [I 0]
    '''

    # Compute the left epipole
    e = compute_epipole_from_F(F,left=True)

    Te = skew(e)
    P = np.vstack((np.dot(Te,F.T).T,e)).T
    return P


def solve_PnP(x,X):

    n = x.shape[1]
    M = np.zeros((3*n,12+n))
    for i in range(n):
        M[3*i,0:4] = X[:,i]
        M[3*i+1,4:8] = X[:,i]
        M[3*i+2,8:12] = X[:,i]
        M[3*i:3*i+3,i+12] = -x[:,i]
    U,S,V = np.linalg.svd(M)
    return V[-1,:12].reshape((3,4))


def PnP(x,X):
    x_homo, X_homo = x,X
    x,X = x[:2], X[:3]
    num = x.shape[1]

    x_mean = np.mean(x,axis=1)
    X_mean = np.mean(X,axis=1)
    x_scale = np.mean(np.sqrt(sum((x-x_mean.reshape(-1,1))**2))) / np.sqrt(2)
    X_scale = np.mean(np.sqrt(sum((x-x_mean.reshape(-1,1))**2))) / np.sqrt(3)

    T = np.linalg.inv(np.array([[x_scale,0,x_mean[0]],[0,x_scale,x_mean[1]],[0,0,1]]))
    U = np.linalg.inv(np.array([[X_scale,0,0,X_mean[0]],[0,X_scale,0,X_mean[1]],[0,0,X_scale,X_mean[2]],[0,0,0,1]]))
    x_n = np.dot(T,x_homo)
    X_n = np.dot(U,X_homo)

    A = np.vstack((np.concatenate((X_n.T,np.zeros((num,4)),(X_n*-x_n[0]).T),axis=1),
                   np.concatenate((np.zeros((num,4)),-X_n.T,(X_n*x_n[1]).T),axis=1)))

    u,s,v = np.linalg.svd(A)
    P = np.dot(np.dot(np.linalg.inv(T),v[-1].reshape((3,4))),U)
    return P


def solve_PnP_Ransac(x,X,threshold=10):

    def PnP_handle(data,*param):
        x = data[:3]
        X = data[3:]
        P = solve_PnP(x,X)
        # P = PnP(x,X)
        return np.ravel(P)

    def PnP_error(model,data,*param):
        x_true = data[:3]
        X = data[3:]
        P = model.reshape((3,4))

        x_cal = np.dot(P,X)
        x_cal /= x_cal[-1]
        return reprojection_error(x_true,x_cal)

    data = np.vstack((x,X))
    result = ransac.loRansacSimple(PnP_handle,PnP_error,data,6,threshold=threshold,maxIter=500)

    return result['model'].reshape((3,4)), result['inliers']


def focal_length_from_F(F):
    e1 = compute_epipole_from_F(F)
    e2 = compute_epipole_from_F(F.T)

    e1_rot = np.array([np.sqrt(e1[0]**2+e1[1]**2), 0, e1[2]])
    e2_rot = np.array([np.sqrt(e2[0]**2+e2[1]**2), 0, e2[2]])

    phi_1 = math.acos(np.dot(e1[:2],e1_rot[:2]) / (e1[0]**2+e1[1]**2)) * ((e1[1]<0)*2-1)
    phi_2 = math.acos(np.dot(e2[:2],e2_rot[:2]) / (e2[0]**2+e2[1]**2)) * ((e2[1]<0)*2-1)

    T1 = np.array([[math.cos(phi_1), -math.sin(phi_1), 0], [math.sin(phi_1), math.cos(phi_1), 0], [0,0,1]])
    T2 = np.array([[math.cos(phi_2), -math.sin(phi_2), 0], [math.sin(phi_2), math.cos(phi_2), 0], [0,0,1]])

    F_new = np.dot(np.dot(np.linalg.inv(T2).T, F), np.linalg.inv(T1))
    D_1 = np.diag(np.array([e1_rot[2],1,-e1_rot[0]]))
    D_2 = np.diag(np.array([e2_rot[2],1,-e2_rot[0]]))

    conic = np.dot(np.dot(np.linalg.inv(D_2), F_new), np.linalg.inv(D_1))

    a,b,c,d = conic[0,0],conic[0,1],conic[1,0],conic[1,1]
    k1 = np.sqrt(-a*c*e1_rot[0]**2 / (a*c*e1_rot[2]**2+b*d))
    k2 = np.sqrt(-a*b*e2_rot[0]**2 / (a*b*e2_rot[2]**2+c*d))
    
    # print('a={}, b={}, c={}, d={}\n'.format(a,b,c,d))
    return k1,k2


def focal_length_from_F_and_P(F,p1,p2):
    '''
    This function computes the focal length corresponding to the principle point p1

    To get the other focal length, interchange the parameter into (F.T, p2, p1)
    '''

    I = np.diag([1,1,0])
    e2 = compute_epipole_from_F(F,left=True)

    num = np.linalg.multi_dot([p2,skew(e2),I,F,p1]) * np.linalg.multi_dot([p1,F.T,p2])
    denom = np.linalg.multi_dot([p2,skew(e2),I,F,I,F.T,p2])

    return -num/denom


def focal_length_iter(x1,x2,p1,p2,f1,f2):

    # Ensure F has rank 2
    def F_rank2(x1,x2,p1,p2,f1,f2):
        F, mask = computeFundamentalMat(x1,x2,error=5)
        K1 = np.array([[f1,0,p1[0]],[0,f1,p1[1]],[0,0,1]])
        K2 = np.array([[f2,0,p2[0]],[0,f2,p2[1]],[0,0,1]])

        E = np.dot(np.dot(K2.T,F),K1)
        U,S,V = np.linalg.svd(E)
        S[0], S[1], S[2] = 1, 1, 0
        E = np.dot(np.dot(U,np.diag(S)),V)
        F = np.dot(np.dot(np.linalg.inv(K2).T,E),np.linalg.inv(K1))
        return F


    # Define cost function
    def focal_length_cost(M,data,guess):
        # Decompose model, data and guess
        F, p1, p2 = M[:9].reshape((3,3)), np.append(M[9:11],1), np.append(M[11:],1)
        x1, x2 = data[:3], data[3:]
        p1_g, p2_g, f1_g, f2_g = guess[:3], guess[3:6], guess[6]**2, guess[7]**2

        # Cost 1
        c1 = np.sum(Sampson_error(x1,x2,F))

        # Cost 2
        w2 = 0.01
        c2_1 = (p1[0]-p1_g[0])**2 + (p1[1]-p1_g[1])**2
        c2_2 = (p2[0]-p2_g[0])**2 + (p2[1]-p2_g[1])**2

        # Cost 3
        w3 = 0
        f1 = focal_length_from_F_and_P(F,p1,p2)
        f2 = focal_length_from_F_and_P(F.T,p2,p1)
        c3_1 = (f1-f1_g)**2
        c3_2 = (f2-f2_g)**2

        # Cost dependent
        w4 = 0.01
        f_min = 100*2
        d1 = f_min > f1
        d2 = f_min > f2
        c4_1 = (f1-f_min)**2
        c4_2 = (f2-f_min)**2

        cost = c1 + w2**2*(c2_1+c2_2) + w3**2*(c3_1+c3_2) + w3**2*(c4_1*d1+c4_2*d2)
        return cost


    # Initial values
    # F_ini = F_rank2(x1,x2,p1,p2,f1,f2)
    # F_ini /= F_ini[2,2]
    F_ini, mask = computeFundamentalMat(x1,x2,error=5)
    M_ini = np.concatenate((np.ravel(F_ini),p1[:2],p2[:2]),axis=0)
    guess = np.concatenate((p1,p2,np.array([f1,f2])),axis=0)

    # Least square optimization
    data = np.vstack((x1,x2))
    fn = lambda x: focal_length_cost(x,data,guess)
    res = least_squares(fn,M_ini)
    M_o = res["x"]

    F_o, p1_o, p2_o = M_o[:9].reshape((3,3)), np.append(M_o[9:11],1), np.append(M_o[11:],1)
    f1_o = focal_length_from_F_and_P(F_o, p1_o, p2_o)
    f2_o = focal_length_from_F_and_P(F_o.T, p2_o, p1_o)

    return f1_o, f2_o


def triangulate_point(x1,x2,P1,P2):
    '''
    Triangulate a single point using least square (SVD)
    '''

    M = np.zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2

    U,S,V = np.linalg.svd(M)
    X = V[-1,:4]

    return X / X[3]


def triangulate(x1,x2,P1,P2):
    '''
    Triangulate multiple points, x1 and x2 in form of (3*N)
    '''

    X = [triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(x1.shape[1])]
    return np.array(X).T


def triangulate_matlab(x1,x2,P1,P2):

    X = np.zeros((4,x1.shape[1]))
    for i in range(x1.shape[1]):
        r1 = x1[0,i]*P1[2] - P1[0]
        r2 = x1[1,i]*P1[2] - P1[1]
        r3 = x2[0,i]*P2[2] - P2[0]
        r4 = x2[1,i]*P2[2] - P2[1]

        A = np.array([r1,r2,r3,r4])
        U,S,V = np.linalg.svd(A)
        X[:,i] = V[-1]/V[-1,-1]
    
    return X
    

def compute_Rt_from_E(E):
    '''
    Compute the camera matrix P2, where P1=[I 0] assumed

    Return 4 possible combinations of R and t as a list
    '''

    U,S,V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U,V))<0:
        V = -V
    E = np.dot(U,np.dot(np.diag([1,1,0]),V))

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R1 = np.dot(U,np.dot(W,V))
    R2 = np.dot(U,np.dot(W.T,V))
    R1 = R1 * np.linalg.det(R1)
    R2 = R2 * np.linalg.det(R2)
    t1 = U[:,2].reshape((-1,1))
    t2 = -U[:,2].reshape((-1,1))

    Rt = [np.hstack((R1,t1)),np.hstack((R1,t2)),np.hstack((R2,t1)),np.hstack((R2,t2))]    

    # P2 = [np.vstack((np.dot(U,np.dot(W,V)).T,U[:,2])).T,
    #       np.vstack((np.dot(U,np.dot(W,V)).T,-U[:,2])).T,
    #       np.vstack((np.dot(U,np.dot(W.T,V)).T,U[:,2])).T,
    #       np.vstack((np.dot(U,np.dot(W.T,V)).T,-U[:,2])).T]
    return Rt


def triangulate_from_E_old(E,K1,K2,x1,x2):
    '''
    Not correct !! 

    Use "triangulate_from_E" instead
    '''

    infront_max = 0

    Rt = compute_Rt_from_E(E)
    P1 = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    for i in range(4):
        P2_temp = np.dot(K2,Rt[i])
        X = triangulate_matlab(x1,x2,P1,P2_temp)
        d1 = np.dot(P1,X)[2]
        d2 = np.dot(P2_temp,X)[2]

        if sum(d1>0)+sum(d2>0) > infront_max:
            infront_max = sum(d1>0)+sum(d2>0)
            infront = (d1>0) & (d2>0)
            P2 = P2_temp
    
    X = triangulate_matlab(x1,x2,P1,P2) 
    return X[:,:], P2


def triangulate_from_E(E,K1,K2,x1,x2):
    x1n = np.dot(np.linalg.inv(K1),x1)
    x2n = np.dot(np.linalg.inv(K2),x2)

    infront_max = 0

    Rt = compute_Rt_from_E(E)
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    for i in range(4):
        P2_temp = Rt[i]
        X = triangulate_matlab(x1n,x2n,P1,P2_temp)
        d1 = np.dot(P1,X)[2]
        d2 = np.dot(P2_temp,X)[2]

        if sum(d1>0)+sum(d2>0) > infront_max:
            infront_max = sum(d1>0)+sum(d2>0)
            infront = (d1>0) & (d2>0)
            P2 = P2_temp
    
    X = triangulate_matlab(x1n,x2n,P1,P2) 
    return X[:,:], P2


def triangulate_cv(E,K1,K2,x1,x2):
    '''
    Triangulation with OpenCV functions
    '''

    num, R, t, mask = cv2.recoverPose(E, x1[:2].T, x2[:2].T, K1)
    P1 = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    P2 = np.dot(K2,np.hstack((R,t)))

    # x1 = np.dot(np.linalg.inv(K1),x1)
    # x2 = np.dot(np.linalg.inv(K2),x2)
    # P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    # P2 = np.hstack((R,t))

    pts1 = x1[:2].astype(np.float64)
    pts2 = x2[:2].astype(np.float64)
    X = cv2.triangulatePoints(P1,P2,pts1,pts2)
    return X/X[-1], np.hstack((R,t))


def undistort(x_homo,coeff):

    # Method 1
    # x_dist = x_homo[:2]

    # def dist_model(x):
    #     u,v = np.split(x,2)
    #     r = u**2 + v**2
    #     res_1 = u + u*coeff[0]*r + u*coeff[1]*r**2 - x_dist[0]
    #     res_2 = v + v*coeff[0]*r + v*coeff[1]*r**2 - x_dist[1]
    #     return np.concatenate((res_1,res_2))

    # sol = root(dist_model,x_dist.ravel())
    # x_homo[:2] = sol.x.reshape((2,-1))

    # Method 2
    def dist_model(x,*arg):
        c1, c2, x_dist, y_dist = arg[0], arg[1], arg[2], arg[3]
        r = x[0]**2 + x[1]**2
        return [x[0]*(1+c1*r+c2*r**2)-x_dist,x[1]*(1+c1*r+c2*r**2)-y_dist]

    for i in range(x_homo.shape[1]):
        sol = root(dist_model,[x_homo[0,i],x_homo[1,i]],args=(coeff[0],coeff[1],x_homo[0,i],x_homo[1,i]))
        x_homo[:2,i] = sol.x

    return x_homo


def reprojection_error(x,x_p):
    return np.sqrt((x[0]-x_p[0])**2 + (x[1]-x_p[1])**2)


if __name__ == "__main__":

    ''' Test '''
    img1 = cv2.imread('C:/Users/tong2/MyStudy/ETH/2018HS/ComputerVision/lab/lab04/cv_lab04_model_fitting/src/epipolar_geometry/images/pumpkin1.jpg',0)
    img2 = cv2.imread('C:/Users/tong2/MyStudy/ETH/2018HS/ComputerVision/lab/lab04/cv_lab04_model_fitting/src/epipolar_geometry/images/pumpkin2.jpg',0)

    # Extract SIFT features
    kp1, des1 = extract_SIFT_feature(img1)
    kp2, des2 = extract_SIFT_feature(img2)

    # Match features
    pts1, pts2, matches, matchesMask = matching_feature(kp1, kp2, des1, des2, ratio=0.8)

    # Compute fundametal matrix F1
    F1, mask = computeFundamentalMat(pts1, pts2)
    pts1 = np.int32(pts1)[mask.ravel()==1]
    pts2 = np.int32(pts2)[mask.ravel()==1]

    # Compute fundametal matrix F2 using inliers
    x1 = np.vstack((pts1.T,np.ones(pts1.shape[0])))
    x2 = np.vstack((pts2.T,np.ones(pts2.shape[0])))
    F2 = compute_fundamental(x1,x2)

    # # Draw epipolar lines
    vis.plotEpiline(img1, img2, pts1, pts2, F1)

    # Draw epipolar lines
    vis.plot_epipolar_line(img1,img2,F2,x1,x2)
