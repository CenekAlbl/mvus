# Classes that are common to the entire project
import numpy as np
import util
import epipolar as ep
import synchronization
import scipy.io as scio
import pickle
import argparse
import copy
import cv2
import visualization as vis
from datetime import datetime
from scipy.optimize import least_squares
from scipy import interpolate


class Scene:
    """ 
    Class that contains everything that we know about the scene, both measured and calculated data

    This class will contain all data and will be passed around to be changed and used by most methods. 

    Members
    -------
    cameras : list of elements of class Camera 
    detections : list of elements of class TrackedObject

    Methods
    -------

    """

    def __init__(self):
        """ 
        Default constructor, creates a Scene with no data
        """
        self.cameras = []
        self.detections = []
        self.detections_raw = []
        self.tracks = []
        self.numCam = 0
        self.beta = None
        self.traj = None
        self.sequence = None
        self.spline = None
        self.visible = []
        self.detections_undist = []
        self.detections_smooth = []


    def addCamera(self,*camera):
        """
        Adds one or more cameras to the scene, first it checks whether the passed object is an instance of Camera

        Parameters
        ----------
        camera : Camera
            the camera to be added
        """
        for i in camera:
            assert type(i) is Camera, "camera is not an instance of Camera"
            self.cameras.append(i)
            self.numCam += 1


    def addDetection(self,*detection):
        """
        Adds one or more detections to the scene.

        Parameters
        ----------
        detection : 
            2D detection in form of (frameId,x,y)*N
        """
        for i in detection:
            assert i.shape[0]==3, "Detection must in form of (x,y,frameId)*N"
            self.detections.append(i)


    def addDetectionRaw(self,*detection_raw):

        for i in detection_raw:
            assert i.shape[0]==3, "Detection must in form of (x,y,frameId)*N"
            self.detections_raw.append(i)

    
    def compute_beta(self,d_min=-6,d_max=6,threshold_1=10,threshold_2=2,threshold_error=3,spline=False):
        '''
        This function computes the pairwise time shift beta, currently using brute-force solver

        The computation will be terminated, when a consistent solution is found (beta_error < threshold_error)
        '''

        self.beta = np.zeros((self.numCam, self.numCam))
        while True:
            beta_error = 0
            for i in range(self.numCam-1):
                for j in range(i+1,self.numCam):
                    if spline:
                        d1 = util.homogeneous(self.detections_smooth[i][1:])
                        d2 = util.homogeneous(self.detections_smooth[j][1:])
                    else:
                        d1 = util.homogeneous(self.detections_undist[i][1:])
                        d2 = util.homogeneous(self.detections_undist[j][1:])
                    numPoints = min(d1.shape[1], d2.shape[1])

                    param = {'k':1, 's':0}
                    self.beta[i,j], F, inliers = synchronization.search_sync(d1[:,:numPoints], d2[:,:numPoints], 
                                                 param=param, d_min=d_min, d_max=d_max, threshold1=threshold_1, threshold2=threshold_2)
                    self.beta[j,i] = -self.beta[i,j]
                beta_error += self.beta[i,i+1]
            beta_error -= self.beta[0,-1]

            print('\nBeta error: {}\n'.format(beta_error))
            print(self.beta)

            if abs(beta_error) < threshold_error:
                break

    
    def undistort_detections(self,apply=True):
        '''
        Undistort images according to calibration, or do nothing but copy the original detections
        '''

        if apply:
            self.detections_undist = detect_undistort(self.detections,self.cameras)
        else:
            self.detections_undist = copy.deepcopy(self.detections)


    def init_spline(self,*arg):
        '''
        Initialize the smooth parameter for each detection as the number of points
        '''
        if len(arg):
            self.spline = arg[0]
        else:
            self.spline = np.array([])
            for i in self.detections_undist:
                self.spline = np.append(self.spline,i.shape[1])

        self.detections_smooth = detect_spline_fitting(self.detections_undist,self.spline)


    def set_sequence(self,*arg):
        '''Set the order for incremental reconstruction according to beta or customizing'''
        if len(arg):
            self.sequence = arg[0]
        else:
            # self.sequence = self.beta[0].argsort()[::-1]
            
            a = np.array([self.tracks[0][0,0]])
            for i in range(1,self.numCam):
                a = np.append(a,self.tracks[i][0,0])
            self.sequence = a.argsort()


    def set_tracks(self,auto=True,*beta,spline=False):
        '''
        This function transfer detections into tracks, which are temporally aligned in a global timeline.

        It will be applied to all detections.
        '''

        if not len(beta):
            beta = self.beta[0]
        else:
            beta = beta[0]

        if spline:
            Track_all = detect_to_track(self.detections_smooth,beta)
        else:
            Track_all = detect_to_track(self.detections_undist,beta)

        if auto:
            self.tracks = Track_all
            return Track_all
        else:
            return Track_all
    

    def set_visibility(self):
        '''This function gives the binary visibility (0 or 1) for self.tracks according to self.traj'''
        v = np.zeros((self.numCam,self.traj.shape[1]))
        for i in range(self.numCam):
            inter,idx1,idx2 = np.intersect1d(self.traj[0],self.tracks[i][0],assume_unique=True,return_indices=True)
            v[i,idx1] = 1
        self.visible = v


    def fit_spline(self,idx=[],s=0.01):

        if len(idx):
            self.spline, _ = interpolate.splprep(self.traj[1:,idx],u=self.traj[0,idx],s=s)
        else:
            self.spline, _ = interpolate.splprep(self.traj[1:],u=self.traj[0],s=s)

        self.spline[1] = np.asarray(self.spline[1])
        self.traj[1:]  = interpolate.splev(self.traj[0],self.spline)


    def init_traj(self,error=1,F=True,inlier_only=False):
        '''
        Select the first two tracks in the sequence, compute fundamental matrix, triangulate points

        It can be chosen: 
                            1. use F or E
                            2. correct matches for polynomial triangulation or not

        Best threshold: 25 for F, 0.006 for E
        '''

        t1, t2 = self.sequence[0], self.sequence[1]
        K1, K2 = self.cameras[t1].K, self.cameras[t2].K
        inter,idx1,idx2 = np.intersect1d(self.tracks[t1][0],self.tracks[t2][0],assume_unique=True,return_indices=True)
        track_1, track_2 = self.tracks[t1][:,idx1], self.tracks[t2][:,idx2]
        
        if F:
            # OpenCV fundamental
            F,inlier = ep.computeFundamentalMat(track_1[1:],track_2[1:],error=error)
            E = np.dot(np.dot(K2.T,F),K1)
        else:
            # Opencv essential
            E,inlier = ep.computeEssentialMat(util.homogeneous(track_1[1:]),util.homogeneous(track_2[1:]),K1,K2,error=error)
            F = np.dot(np.linalg.inv(K2.T),np.dot(E,np.linalg.inv(K1)))

        # Own implementation
        # res = ep.compute_fundamental_Ransac(util.homogeneous(track_1[1:]),util.homogeneous(track_2[1:]),threshold=error,loRansac=False)
        # F = res['model'].reshape(3,3)
        # inlier = np.zeros(track_1.shape[1]).astype(int)
        # inlier[res['inliers']] = 1
        # E = np.dot(np.dot(K2.T,F),K1)

        if not inlier_only:
            inlier = np.ones(len(inlier))
        x1, x2 = util.homogeneous(track_1[1:,inlier==1]), util.homogeneous(track_2[1:,inlier==1])

        # Find corrected corresponding points for optimal triangulation
        N = track_1[1:,inlier==1].shape[1]
        pts1=track_1[1:,inlier==1].T.reshape(1,-1,2)
        pts2=track_2[1:,inlier==1].T.reshape(1,-1,2)
        m1,m2 = cv2.correctMatches(F,pts1,pts2)
        x1,x2 = util.homogeneous(np.reshape(m1,(-1,2)).T), util.homogeneous(np.reshape(m2,(-1,2)).T)

        X, P = ep.triangulate_from_E(E,K1,K2,x1,x2)
        self.traj = np.vstack((inter[inlier==1],X[:-1]))

        # # plot epipolar lines
        # img1, img2  = np.zeros((1080,1920),dtype=np.uint8), np.zeros((1080,1920),dtype=np.uint8)
        # x1_int, x2_int = np.int16(x1[:2]), np.int16(x2[:2])

        # for i in range(x1.shape[1]):
        #     img1[x1_int[1,i],x1_int[0,i]]= 255
        # for i in range(x2.shape[1]):
        #     img2[x2_int[1,i],x2_int[0,i]]= 255

        # idx = np.random.choice(x1.shape[1],30)
        # vis.plot_epipolar_line(img1, img2, F, x1[:,idx], x2[:,idx])

        # Assign the camera matrix for these two cameras
        self.cameras[t1].P = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
        self.cameras[t2].P = np.dot(K2,P)
        self.cameras[t1].decompose()
        self.cameras[t2].decompose()

        return idx1[inlier==1],idx2[inlier==1]


    def get_camera_pose(self,id_cam,error=8):
        '''
        This function solve the PnP problem for a single camera with fixed points.

        Currently using OpenCV function solvePnPRansac, which exploits Ransac and LM-optimization
        '''

        inter,idx1,idx2 = np.intersect1d(self.traj[0],self.tracks[id_cam][0],assume_unique=True,return_indices=True)
        X = util.homogeneous(self.traj[1:,idx1])
        x = util.homogeneous(self.tracks[id_cam][1:,idx2])
        x_n = np.dot(np.linalg.inv(self.cameras[id_cam].K), x)

        # OpenCV
        N = X.shape[1]
        objectPoints = np.ascontiguousarray(X[:3].T).reshape((N,1,3))
        imagePoints  = np.ascontiguousarray(x[:2].T).reshape((N,1,2))
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints,imagePoints,self.cameras[id_cam].K,None,reprojectionError=error)

        # Own implementation
        # P,inlier = ep.solve_PnP_Ransac(x_n,X,threshold=thres)
        # Cam_temp = Camera(P=np.dot(self.cameras[id_cam].K, P))
        # # P,inlier = ep.solve_PnP_Ransac(x,X,threshold=8)
        # # Cam_temp = Camera(P=P)

        # self.traj = self.traj[:,idx1[inliers.reshape(-1)]]

        self.cameras[id_cam].R = cv2.Rodrigues(rvec)[0]
        self.cameras[id_cam].t = tvec.reshape(-1,)
        self.cameras[id_cam].compose()

        print('Number of inliers for PnP: {}'.format(inliers.shape[0]))

    
    def error_cam(self,id_cam,thres=0,dist=True):
        '''
        This function computes the reprojection error for a single camera, such that only visible 3D points will be reprojected
        '''

        inter,idx1,idx2 = np.intersect1d(self.traj[0],self.tracks[id_cam][0],assume_unique=True,return_indices=True)
        X = util.homogeneous(self.traj[1:,idx1])
        x = util.homogeneous(self.tracks[id_cam][1:,idx2])
        x_cal = self.cameras[id_cam].projectPoint(X)

        if dist:
            error = ep.reprojection_error(x,x_cal)
        else:
            error = np.concatenate((abs(x_cal[0]-x[0]),abs(x_cal[1]-x[1])))
        
        print("Mean reprojection error for camera {} is: ".format(id_cam), np.mean(error))

        # Optional: remove large errors directly
        if thres:
            self.traj = np.delete(self.traj,idx1[error>thres],axis=1)
            self.error_cam(id_cam)
        else:
            return error


    def triangulate_traj(self,cam1,cam2,overwrite=False):
        '''
        This function triangulates points from tracks in two cameras.

        These 3D points can either overwrite the original trajectory, or be appended to it
        '''

        inter,idx1,idx2 = np.intersect1d(self.tracks[cam1][0],self.tracks[cam2][0],assume_unique=True,return_indices=True)
        track_1, track_2 = self.tracks[cam1][:,idx1], self.tracks[cam2][:,idx2]
        x1, x2 = util.homogeneous(track_1[1:]), util.homogeneous(track_2[1:])
        P1, P2 = self.cameras[cam1].P, self.cameras[cam2].P
        X = ep.triangulate_matlab(x1,x2,P1,P2)
        X_idx = np.vstack((inter,X[:-1]))

        if overwrite:
            self.traj = X_idx
        else:
            _,idx1,_ = np.intersect1d(inter,self.traj[0],assume_unique=True,return_indices=True)
            idx_new = np.setdiff1d(np.arange(len(inter)),idx1,assume_unique=True)
            if len(idx_new):
                X_new = np.hstack((self.traj,X_idx[:,idx_new]))
                self.traj = X_new[:,np.argsort(X_new[0])]
            else:
                print('No new points are triangulated')
                
        return X_idx

                
class Camera:
    """ 
    Class that describes a single camera in the scene

    This class contains parameters of a single camera in the scene, i.e. its calibration parameters, pose parameters and its images
    
    Members
    -------
    K : calibration matrix
    R : camera orientation
    t : camera center
    d : distortion coefficients

    Methods
    -----
    projectPoint: get 2D coords from x=PX
    decompose: decompose P into K,R,t
    center: acquire 3D coords of the camera center

    """

    def __init__(self,**kwargs):
        self.P = kwargs.get('P')
        self.K = kwargs.get('K')
        self.R = kwargs.get('R')
        self.t = kwargs.get('t')
        self.d = kwargs.get('d')
        self.c = kwargs.get('c')


    def projectPoint(self,X):
        x = np.dot(self.P,X)
        x /= x[2]
        return x


    def compose(self):
        self.P = np.dot(self.K,np.hstack((self.R,self.t.reshape((-1,1)))))


    def decompose(self):
        M = self.P[:,:3]
        R,K = np.linalg.qr(np.linalg.inv(M))
        R = np.linalg.inv(R)
        K = np.linalg.inv(K)

        T = np.diag(np.sign(np.diag(K)))
        if np.linalg.det(T) < 0:
            T[1,1] *= -1

        self.K = np.dot(K,T)
        self.R = np.dot(T,R)
        self.t = np.dot(np.linalg.inv(self.K),self.P[:,3])
        self.K /= self.K[-1,-1]

        return self.K, self.R, self.t


    def center(self):
        if self.c is not None:
            return self.c
        else:
            self.decompose()
            self.c = -np.dot(self.R.T,self.t)
            return self.c


    def P2vector(self,n=6):
        '''
        Convert camera parameters into a vector
        '''

        k = np.array([(self.K[0,0]+self.K[1,1])/2, self.K[0,2], self.K[1,2]])
        r = cv2.Rodrigues(self.R)[0].reshape(-1,)

        if n==6:
            return np.concatenate((r,self.t))
        elif n==8:
            return np.concatenate((r,self.t,self.d))
        elif n==9:
            return np.concatenate((k,r,self.t))
        elif n==11:
            return np.concatenate((k,r,self.t,self.d))
        elif n==12:
            k = np.array([self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]])
            return np.concatenate((k,r,self.t,self.d))


    def vector2P(self,vector,n=6):
        '''
        Convert a vector into camera parameters
        '''

        if n==6:
            self.R = cv2.Rodrigues(vector[:3])[0]
            self.t = vector[3:6]
        elif n==8:
            self.R = cv2.Rodrigues(vector[:3])[0]
            self.t = vector[3:6]
            self.d = vector[-2:]
        elif n==9:
            self.K = np.diag((1,1,1)).astype(float)
            self.K[0,0], self.K[1,1] = vector[0], vector[0]
            self.K[:2,-1] = vector[1:3]
            self.R = cv2.Rodrigues(vector[3:6])[0]
            self.t = vector[6:9]
        elif n==11:
            self.K = np.diag((1,1,1)).astype(float)
            self.K[0,0], self.K[1,1] = vector[0], vector[0]
            self.K[:2,-1] = vector[1:3]
            self.R = cv2.Rodrigues(vector[3:6])[0]
            self.t = vector[6:9]
            self.d = vector[-2:]
        elif n==12:
            self.K = np.diag((1,1,1)).astype(float)
            self.K[0,0], self.K[1,1] = vector[0], vector[1]
            self.K[:2,-1] = vector[2:4]
            self.R = cv2.Rodrigues(vector[4:7])[0]
            self.t = vector[7:10]
            self.d = vector[-2:]

        self.compose()
        return self.P
    

    def info(self):
        print('\n P:')
        print(self.P)
        print('\n K:')
        print(self.K)
        print('\n R:')
        print(self.R)
        print('\n t:')
        print(self.t)


def detect_undistort(d,cameras):
    '''
    Use Opencv undistpoints

    Input detection has the form [frame_id, x, y]
    '''

    detections = copy.deepcopy(d)
    for i in range(len(detections)):
            src = np.ascontiguousarray(detections[i][1:].T).reshape((detections[i].shape[1],1,2))
            dst = cv2.undistortPoints(src,cameras[i].K,np.concatenate((cameras[i].d,np.array([0,0]))))
            dst_unnorm = np.dot(cameras[i].K, util.homogeneous(dst.reshape((detections[i].shape[1],2)).T))
            detections[i][1:] = dst_unnorm[:2]

    # for i in range(len(detections)):
    #         src = np.ascontiguousarray(detections[i][1:].T).reshape((detections[i].shape[1],1,2))
    #         dst = cv2.undistortPoints(src,cameras[i].K,np.concatenate((cameras[i].d,np.array([0,0]))))
    #         dst_unnorm = np.dot(cameras[i].K, util.homogeneous(dst.reshape((detections[i].shape[1],2)).T))
    #         detections[i][1:] = dst_unnorm[:2]

    return detections


def detect_spline_fitting(d,smooth):
    detections = copy.deepcopy(d)
    for i in range(len(detections)):
        detections[i][1] = util.spline_fitting(detections[i][1],detections[i][0],detections[i][0],k=3,s=smooth[i])
        detections[i][2] = util.spline_fitting(detections[i][2],detections[i][0],detections[i][0],k=3,s=smooth[i])
    return detections


def detect_to_track(detections,beta):

    # Track_all = [[] for i in range(len(detections))]
    # idx_min = 0
    # for i in range(len(detections)):
    #     track = copy.deepcopy(detections[i])
    #     track[0] -= track[0,0]

    #     if i == 0:
    #         Track_all[i] = track
    #     else:
    #         track_idx = track[0] - beta[i]
    #         track[1] = util.spline_fitting(track[1],np.round(track_idx),track_idx,k=1,s=0)
    #         track[2] = util.spline_fitting(track[2],np.round(track_idx),track_idx,k=1,s=0)
    #         track[0] = np.round(track_idx)
    #         Track_all[i] = track[:,1:-1]

    #         if Track_all[i][0,0] < idx_min:
    #             idx_min = Track_all[i][0,0]
    # for i in Track_all:
    #     i[0] = i[0] - idx_min

    Track_all = [[] for i in range(len(detections))]
    for i in range(len(detections)):
        track = copy.deepcopy(detections[i])
        if i == 0:
            Track_all[i] = track
            idx_min = track[0,0]
        else:
            track_idx = track[0] - beta[i]
            track[1] = util.spline_fitting(track[1],np.round(track_idx),track_idx,k=1,s=0)
            track[2] = util.spline_fitting(track[2],np.round(track_idx),track_idx,k=1,s=0)
            track[0] = np.round(track_idx)
            Track_all[i] = track[:,1:-1]

            if Track_all[i][0,0] < idx_min:
                idx_min = Track_all[i][0,0]
    for i in Track_all:
        i[0] = i[0] - idx_min

    return Track_all


def jac_twoCam(N,n_cam,M=0):
    '''
    N: number of points
    n_cam: number of camera parameters
    M: number of spline coefficients
    '''
    if M:
        jac_X = np.ones((4*N,M))
    else:
        jac_X = np.zeros((4*N,3*N))
        for i in range(N):
            jac_X[(i,i+N,i+2*N,i+3*N),i*3:(i+1)*3] = 1

    jac_cam = np.zeros((4*N,2*n_cam))
    jac_cam[:2*N,:n_cam] = 1
    jac_cam[2*N:,n_cam:] = 1

    return np.hstack((jac_cam,jac_X))


def optimize_two(c1,c2,x1,x2,traj,spline=[],include_K=False,max_iter=10):

    def error_fn(model):
        c1.vector2P(model[:n_cam],n=n_cam)
        c2.vector2P(model[n_cam:2*n_cam],n=n_cam)

        if len(spline):
            spline[1] = model[2*n_cam:].reshape(3,-1)
            x,y,z = interpolate.splev(traj[0],spline)
            X = np.array([x,y,z])
        else:
            X = model[2*n_cam:].reshape(-1,3).T

        x_c1 = c1.projectPoint(util.homogeneous(X))
        x_c2 = c2.projectPoint(util.homogeneous(X))

        return np.concatenate((abs(x1[0]-x_c1[0]),abs(x1[1]-x_c1[1]),abs(x2[0]-x_c2[0]),abs(x2[1]-x_c2[1])))
    
    # before
    if include_K: 
        n_cam = 9 
    else: 
        n_cam = 6

    if len(spline):
        model = np.concatenate((c1.P2vector(n=n_cam),c2.P2vector(n=n_cam),np.ravel(spline[1])))
        A = jac_twoCam(x1.shape[1],n_cam,spline[1].shape[1]*3)
    else:
        model = np.concatenate((c1.P2vector(n=n_cam),c2.P2vector(n=n_cam),np.ravel(traj[1:].T)))
        A = jac_twoCam(x1.shape[1],n_cam)

    # During
    fn = lambda x: error_fn(x)
    res = least_squares(fn,model,jac_sparsity=A,tr_solver='lsmr',max_nfev=max_iter)

    # After
    c1.vector2P(res.x[:n_cam],n=n_cam)
    c2.vector2P(res.x[n_cam:2*n_cam],n=n_cam)
    if len(spline):
        spline[1] = res.x[2*n_cam:].reshape(3,-1)
        x,y,z = interpolate.splev(traj[0],spline)
        traj[1:] = np.array([x,y,z])
        return res,[c1,c2,traj,spline]
    else:
        traj[1:] = res.x[2*n_cam:].reshape(-1,3).T
        return res,[c1,c2,traj]


def jac_allCam(v,n_cam,beta=[],M=[],N=[]):

    '''
    M: control point indices
    N: trajectory point indices
    '''

    num_cam, num_point = v.shape[0], v.shape[1]

    # 3D points or spline parameters
    if len(M):
        assert len(N)==num_point,'The length of frame indices should equal the number of points'
        jac_X = np.zeros((num_cam*num_point*2,len(M)*3))

        for j in range(num_point):
            c = sum(M<N[j])

            if c==0:
                dep = np.array([0,1])
            elif c==len(M):
                dep = np.array([len(M)-2,len(M)-1])
            else:
                dep = np.array([c-1,c,c+1])
                
            for i in range(num_cam):
                jac_X[i*num_point*2+j,dep] = 1
                jac_X[i*num_point*2+j,dep+len(M)] = 1
                jac_X[i*num_point*2+j,dep+len(M)*2] = 1

                jac_X[i*num_point*2+num_point+j,dep] = 1
                jac_X[i*num_point*2+num_point+j,dep+len(M)] = 1
                jac_X[i*num_point*2+num_point+j,dep+len(M)*2] = 1
        
    else:
        jac_X_i = np.zeros((num_point*2,num_point*3))
        for i in range(num_point):
            jac_X_i[(i,i+num_point),i*3:(i+1)*3] = 1
        jac_X = np.tile(jac_X_i,(num_cam,1))

    # camera
    jac_cam = np.zeros((num_cam*num_point*2,num_cam*n_cam))
    for i in range(num_cam):
        jac_cam[i*num_point*2:(i+1)*num_point*2, i*n_cam:(i+1)*n_cam] = 1

    # beta
    if len(beta):
        assert len(beta)==num_cam, 'The length of beta vector should equal the number of cams'

        jac_beta = np.zeros((num_cam*num_point*2,num_cam))
        for i in range(1,num_cam):
            jac_beta[i*num_point*2:(i+1)*num_point*2,i] = 1

        jac_beta = jac_beta.astype(np.int8)
        jac_cam  = jac_cam.astype(np.int8)
        jac_X    = jac_X.astype(np.int8)

        jac = np.concatenate((jac_beta,jac_cam,jac_X),axis=1)
    else:
        jac_cam  = jac_cam.astype(np.int8)
        jac_X    = jac_X.astype(np.int8)

        jac = np.concatenate((jac_cam,jac_X),axis=1)


    # remove those that are not visible
    for i in range(num_cam):
        for j in range(num_point):
            if v[i,j] == 0:
                jac[(i*num_point*2+j,i*num_point*2+num_point+j),:] = 0

    return jac


def optimize_all(cams,tracks,traj,v,spline,include_K=False,max_iter=10,distortion=False,beta=[]):

    def error_fn(model,Tracks):

        # Decode camera
        for i in range(num_Cam):
            if len(beta):
                cams[i].vector2P(model[num_Cam+i*n_cam:num_Cam+(i+1)*n_cam],n=n_cam)
            else:
                cams[i].vector2P(model[i*n_cam:(i+1)*n_cam],n=n_cam)

        # Decode x and X
        if len(beta):
            if distortion:
                Tracks = detect_undistort(Tracks,cams)
            Tracks = detect_to_track(Tracks,model[:num_Cam])
            if len(spline):
                spline[1] = model[num_Cam+num_Cam*n_cam:].reshape(3,-1)
                x,y,z = interpolate.splev(traj[0],spline)
                traj[1:] = np.array([x,y,z])
            else:
                traj[1:] = model[num_Cam+num_Cam*n_cam:].reshape(-1,3).T
        else:
            if len(spline):
                spline[1] = model[num_Cam*n_cam:].reshape(3,-1)
                x,y,z = interpolate.splev(traj[0],spline)
                traj[1:] = np.array([x,y,z])
            else:
                traj[1:] = model[num_Cam*n_cam:].reshape(-1,3).T

        # Compute error
        error = np.array([])
        for i in range(num_Cam):
            inter,idx1,idx2 = np.intersect1d(traj[0],Tracks[i][0],assume_unique=True,return_indices=True)
            X_temp = util.homogeneous(traj[1:,idx1])
            x = Tracks[i][1:,idx2]
            x_repro = cams[i].projectPoint(X_temp)

            error_cam_i = np.zeros(num_Point*2)
            error_cam_i[idx1],error_cam_i[idx1+num_Point] = abs(x_repro[0]-x[0]),abs(x_repro[1]-x[1])

            error = np.concatenate((error,error_cam_i))
        return error
    
    # before
    n_cam = 6
    if include_K:  n_cam += 3
    if distortion: n_cam += 2
    num_Cam, num_Point = len(cams), traj.shape[1]

    # Set beta
    if len(beta):
        assert len(beta)==num_Cam, 'The length of beta vector should equal the number of cams'
        model = beta
    else:
        model = np.array([])
    
    # Set cameras
    for i in cams:
        model = np.concatenate((model,i.P2vector(n=n_cam)))
    
    # Set 3D points or spline and Jacobian
    print('\nComputing the sparsity structure of the Jacobian matrix...\n')
    if len(spline):
        model = np.concatenate((model,np.ravel(spline[1])))
        A = jac_allCam(v,n_cam,beta=beta,M=spline[0][2:-2],N=traj[0])
    else:
        model = np.concatenate((model,np.ravel(traj[1:].T)))
        A = jac_allCam(v,n_cam,beta=beta)

    # During
    print('Doing BA with {} cameras...\n'.format(num_Cam))
    error_before = error_fn(model,tracks)
    fn = lambda x: error_fn(x,tracks)
    res = least_squares(fn,model,jac_sparsity=A,tr_solver='lsmr',max_nfev=max_iter)

    # After
    if len(beta):
        beta = res.x[:num_Cam]
        for i in range(num_Cam):
            cams[i].vector2P(res.x[num_Cam+i*n_cam:num_Cam+(i+1)*n_cam],n=n_cam)
        if len(spline):
            spline[1] = res.x[num_Cam+num_Cam*n_cam:].reshape(3,-1)
            x,y,z = interpolate.splev(traj[0],spline)
            traj[1:] = np.array([x,y,z])
            return res,[beta,cams,traj,spline]
        else:
            traj[1:] = res.x[num_Cam+num_Cam*n_cam:].reshape(-1,3).T
            return res,[beta,cams,traj]
    else:
        for i in range(num_Cam):
            cams[i].vector2P(res.x[i*n_cam:(i+1)*n_cam],n=n_cam)
        if len(spline):
            spline[1] = res.x[num_Cam*n_cam:].reshape(3,-1)
            x,y,z = interpolate.splev(traj[0],spline)
            traj[1:] = np.array([x,y,z])
            return res,[cams,traj,spline]
        else:
            traj[1:] = res.x[num_Cam*n_cam:].reshape(-1,3).T
            return res,[cams,traj]


if __name__ == "__main__":

    # Load previous scene..
    # with open('./data/jobs/EF_KRt_3cams/optimization_after_F_Rt.pkl', 'rb') as file:
    #     f1 = pickle.load(file)
    # with open('./data/jobs/EF_KRt_3cams/optimization_after_E_Rt.pkl', 'rb') as file:
    #     f2 = pickle.load(file)

    # New scene
    start=datetime.now()
    
    # Load camara intrinsic and radial distortions
    intrin_1 = scio.loadmat('./data/calibration/first_flight/phone_0/calibration.mat')
    intrin_2 = scio.loadmat('./data/calibration/first_flight/phone_1/calibration.mat')
    intrin_3 = scio.loadmat('./data/calibration/first_flight/phone_2/calibration.mat')

    K1, K2, K3 = intrin_1['intrinsic'], intrin_2['intrinsic'], intrin_3['intrinsic']
    d1, d2, d3 = intrin_1['radial_distortion'][0], intrin_2['radial_distortion'][0], intrin_3['radial_distortion'][0]
    cameras = [Camera(K=K1,d=d1), Camera(K=K2,d=d2), Camera(K=K3,d=d3)]

    # Load detections
    detect_1 = np.loadtxt('./data/video_1_output.txt',skiprows=1,dtype=np.int32)
    detect_2 = np.loadtxt('./data/video_2_output.txt',skiprows=1,dtype=np.int32)
    detect_3 = np.loadtxt('./data/video_3_output.txt',skiprows=1,dtype=np.int32)

    # Create a scene
    flight = Scene()
    flight.addCamera(*cameras)
    flight.addDetection(detect_1.T.astype(float), detect_2.T.astype(float), detect_3.T.astype(float))

    # Correct radial distortion, can be set to false
    flight.undistort_detections(apply=True)

    # Initialize spline for detections, decide whether using it
    flight.init_spline()

    # Compute beta for every pair of cameras
    flight.beta = np.array([[0,-81.74,-31.73],[81.74,0,52.25],[31.73,-52.25,0]])
    # flight.compute_beta(threshold_error=2)
    # print('\n',flight.beta,'\n')

    # create tracks according to beta
    flight.set_tracks()

    # Sort detections in temporal order
    flight.set_sequence()

    # Set parameters for initial triangulation through external parsing
    parser = argparse.ArgumentParser(description="Decide whether E or F used and K optimized or not")
    parser.add_argument('-e','-E',help='use E instead of F',action="store_false")
    parser.add_argument('-k','-K',help='Disable optimizing calibration matrix',action="store_false")
    parser.add_argument('-d','-D',help='Disable optimizing radial distortions',action="store_false")
    parser.add_argument('-b','-B',help='Disable optimizing time shifts',action="store_false")
    parser.add_argument('-m','-M',help='Set maximal iteration number for optimization, default is 10',default=10)
    args = vars(parser.parse_args())
    use_F, include_K, include_d, include_b, max_iter = args['e'], args['k'], args['d'], args['b'], args['m']

    # Set parameters manually
    use_F = True
    include_K = True
    include_d = True
    include_b = True
    max_iter = 10
    use_spline = True
    smooth_factor = 0.01

    if use_F:
        E_or_F = 'F'
        error_epip = 25
        error_PnP  = 50
    else:
        E_or_F = 'E'
        error_epip = 0.006
        error_PnP  = 10

    if include_K:
        K_not = ''
        K_or = '_K'
    else:
        K_not = ' not'
        K_or = '_'

    print('\nCurrently using '+E_or_F+', K is'+K_not+' optimized')
    print('Threshold for Epipolar:{}, Threshold for PnP:{}'.format(error_epip,error_PnP))

    # Initialize the first 3D trajectory
    idx1, idx2 = flight.init_traj(error=error_epip,F=use_F,inlier_only=True)

    # Compute spline parameters and smooth the trajectory
    if use_spline:
        flight.fit_spline(s=smooth_factor)
    else:
        flight.spline = []


    '''----------------Optimization----------------'''
    start=datetime.now()

    print('\nBefore optimization:')
    flight.error_cam(0)
    flight.error_cam(2)
    flight_before = copy.deepcopy(flight)

    '''Optimize two'''
    res, model = optimize_two(flight.cameras[0],flight.cameras[2],flight.tracks[0][1:,idx1],
                        flight.tracks[2][1:,idx2],flight.traj,flight.spline,include_K=include_K,max_iter=max_iter)
    flight.cameras[0],flight.cameras[2],flight.traj = model[0], model[1], model[2]
    if use_spline:
        flight.spline = model[3]

    # Check reprojection error
    print('\nAfter optimizating first two cameras:')
    flight.error_cam(0)
    flight.error_cam(2)

    print('\nTime: {}\n'.format(datetime.now()-start))

    vis.show_trajectory_3D(flight.traj[1:],color=False)

    '''Add camera'''
    flight.get_camera_pose(flight.sequence[2],error=error_PnP)
    flight.error_cam(1)

    # Triangulate more points if possible
    flight.triangulate_traj(0,2)
    flight.triangulate_traj(1,2)

    # Fit spline again if needed
    if use_spline:
        flight.fit_spline(s=smooth_factor)
    flight.error_cam(1)

    # Define visibility
    flight.set_visibility()

    '''Optimize all'''
    # Before BA: set parameters
    if include_b:
        beta = flight.beta[0]
        if include_d:
            Track = flight.detections
        else:
            Track = flight.detections_undist
    else:
        include_d = False
        beta = []
        Track = flight.tracks

    # BA
    res, model = optimize_all(flight.cameras,Track,flight.traj,flight.visible,flight.spline,include_K=include_K,
                            max_iter=max_iter,distortion=include_d,beta=beta)

    # After BA: interpret results
    if include_b:
        flight.beta[0], flight.cameras, flight.traj = model[0], model[1], model[2]
    else:
        flight.cameras, flight.traj = model[0], model[1]

    if use_spline:
        flight.spline = model[-1]

    flight.undistort_detections(apply=True)
    flight.set_tracks()

    # Check reprojection error
    print('\nAfter optimazing all cameras, beta:{}, d:{}'.format(include_b,include_d))
    flight.error_cam(0)
    flight.error_cam(1)
    flight.error_cam(2)

    print('\nTime: {}\n'.format(datetime.now()-start))

    print('\nFinished')