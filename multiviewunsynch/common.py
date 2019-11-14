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
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        self.setting = None
        self.gps = None


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


class Scene_multi_spline(Scene):
    '''
    A updated version of the class Scene, in which a global timeline is defined
    '''

    def __init__(self):
        super().__init__()
        self.alpha = None
        self.detections_global = []
        self.spline = {'tck':None, 'int':None}
        self.cam_model = 12


    def init_alpha(self,prior=[]):
        '''Initialize alpha for each camera based on the ratio of fps'''

        if len(prior):
            assert len(prior) == self.numCam, 'Number of input must be the same as the number of cameras'
            self.alpha = prior
        else:
            self.alpha = np.ones(self.numCam)
            fps_ref = self.cameras[self.sequence[0]].fps
            for i in range(self.numCam):
                self.alpha[i] = fps_ref / self.cameras[i].fps


    def detection_to_global(self, *cam, distortion=True):
        '''
        Convert frame indices of raw detections into the global timeline.

        Input is an iterable that specifies which detection(s) to compute.

        If no input, all detections will be converted.
        '''

        assert len(self.alpha)==self.numCam and len(self.beta)==self.numCam, 'The Number of alpha and beta is wrong'

        if not distortion:
            if len(cam):
                for i in cam:
                    detect_global = np.vstack((self.alpha[i] * self.detections[i][0] + self.beta[i], self.detections[i][1:]))
                    self.detections_global[i] = detect_global
            else:
                self.detections_global = []
                for i in range(self.numCam):
                    detect_global = np.vstack((self.alpha[i] * self.detections[i][0] + self.beta[i], self.detections[i][1:]))
                    self.detections_global.append(detect_global)
        else:
            if len(cam):
                for i in cam:
                    timestamp = self.alpha[i] * self.detections[i][0] + self.beta[i]
                    detect = self.cameras[i].undist_point(self.detections[i][1:])
                    self.detections_global[i] = np.vstack((timestamp,detect))
            else:
                self.detections_global = []
                for i in range(self.numCam):
                    timestamp = self.alpha[i] * self.detections[i][0] + self.beta[i]
                    detect = self.cameras[i].undist_point(self.detections[i][1:])
                    self.detections_global.append(np.vstack((timestamp,detect)))


    def cut_detection(self,second=1):
        '''
        Truncate the starting and end part of each continuous part of the detections
        '''

        if not second: return

        for i in range(self.numCam):
            detect = self.detections[i]
            interval = self.find_intervals(detect[0])
            cut = int(self.cameras[i].fps * second)

            interval_long = interval[:,interval[1]-interval[0]>cut*2]
            interval_long[0] += cut
            interval_long[1] -= cut

            assert (interval_long[1]-interval_long[0]>=0).all()

            self.detections[i], _ = self.sampling(detect,interval_long)

    
    def find_intervals(self,x,gap=5,idx=False):
        '''
        Given indices of detections, return a matrix that contains the start and the end of each
        continues part.
        
        Input indices must be in ascending order. 
        
        The gap defines the maximal interruption, with which it's still considered as continues. 
        '''

        assert len(x.shape)==1 and (x[1:]>x[:-1]).all(), 'Input must be an ascending 1D-array'

        # Compute start and end
        x_s, x_e = np.append(-np.inf,x), np.append(x,np.inf)
        start = x_s[1:] - x_s[:-1] >= gap
        end = x_e[:-1] - x_e[1:] <= -gap
        interval = np.array([x[start],x[end]])
        int_idx = np.array([np.where(start)[0],np.where(end)[0]])

        # Remove intervals that are too short
        mask = interval[1]-interval[0] >= gap
        interval = interval[:,mask]
        int_idx = int_idx[:,mask]

        assert (interval[0,1:]>interval[1,:-1]).all()

        if idx:
            return interval, int_idx
        else:
            return interval


    def sampling(self,x,interval,belong=False):
        '''
        Sample points from the input which are inside the given intervals
        '''

        # Define timestamps
        if len(x.shape)==1:
            timestamp = x
        elif len(x.shape)==2:
            assert x.shape[0]==3 or x.shape[0]==4, 'Input should be 1D array or 2D array with 3 or 4 rows'
            timestamp = x[0]

        # Sample points from each interval
        idx_ts = np.zeros_like(timestamp, dtype=int)
        for i in range(interval.shape[1]):
            mask = np.logical_xor(timestamp-interval[0,i] >= 0, timestamp-interval[1,i] >= 0)
            idx_ts[mask] = i+1

        if not belong:
            idx_ts = idx_ts.astype(bool)

        if len(x.shape)==1:
            return x[idx_ts.astype(bool)], idx_ts
        elif len(x.shape)==2:
            return x[:,idx_ts.astype(bool)], idx_ts
        else:
            raise Exception('The shape of input is wrong')


    def match_overlap(self,x,y):
        '''
        Given two inputs in the same timeline (global), return the parts of them which are temporally overlapped

        Important: it's assumed that x has a higher frequency (fps) so that points are interpolated in y
        '''

        interval = self.find_intervals(y[0])
        x_s, _ = self.sampling(x, interval)

        tck, u = interpolate.splprep(y[1:],u=y[0],s=0,k=3)
        y_s = np.asarray(interpolate.splev(x_s[0],tck))
        y_s = np.vstack((x_s[0],y_s))

        assert (x_s[0] == y_s[0]).all(), 'Both outputs should have the same timestamps'

        return x_s, y_s


    def init_traj(self,error=10,inlier_only=False):
        '''
        Select the first two cams in the sequence, compute fundamental matrix, triangulate points
        '''

        t1, t2 = self.sequence[0], self.sequence[1]
        K1, K2 = self.cameras[t1].K, self.cameras[t2].K

        # Find correspondences
        if self.cameras[t1].fps > self.cameras[t2].fps:
            d1, d2 = self.match_overlap(self.detections_global[t1], self.detections_global[t2])
        else:
            d2, d1 = self.match_overlap(self.detections_global[t2], self.detections_global[t1])
        
        # Compute fundamental matrix
        F,inlier = ep.computeFundamentalMat(d1[1:],d2[1:],error=error)
        E = np.dot(np.dot(K2.T,F),K1)

        if not inlier_only:
            inlier = np.ones(len(inlier))
        x1, x2 = util.homogeneous(d1[1:,inlier==1]), util.homogeneous(d2[1:,inlier==1])

        # Find corrected corresponding points for optimal triangulation
        N = d1[1:,inlier==1].shape[1]
        pts1=d1[1:,inlier==1].T.reshape(1,-1,2)
        pts2=d2[1:,inlier==1].T.reshape(1,-1,2)
        m1,m2 = cv2.correctMatches(F,pts1,pts2)
        x1,x2 = util.homogeneous(np.reshape(m1,(-1,2)).T), util.homogeneous(np.reshape(m2,(-1,2)).T)

        mask = np.logical_not(np.isnan(x1[0]))
        x1 = x1[:,mask]
        x2 = x2[:,mask]

        # Triangulte points
        X, P = ep.triangulate_from_E(E,K1,K2,x1,x2)
        self.traj = np.vstack((d1[0][inlier==1][mask],X[:-1]))

        # Assign the camera matrix for these two cameras
        self.cameras[t1].P = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
        self.cameras[t2].P = np.dot(K2,P)
        self.cameras[t1].decompose()
        self.cameras[t2].decompose()


    def traj_to_spline(self,smooth_factor=0.001):
        '''
        Convert discrete 3D trajectory into spline representation

        A single spline is built for each interval
        '''

        timestamp = self.traj[0]
        interval, idx = self.find_intervals(timestamp,idx=True)
        tck = [None] * interval.shape[1]

        for i in range(interval.shape[1]):
            part = self.traj[:,idx[0,i]:idx[1,i]+1]
            try:
                tck[i], u = interpolate.splprep(part[1:],u=part[0],s=smooth_factor,k=3)
            except:
                tck[i], u = interpolate.splprep(part[1:],u=part[0],s=smooth_factor,k=1)
            
        self.spline['tck'], self.spline['int'] = tck, interval
        return self.spline


    def spline_to_traj(self,sampling_rate=1,t=None):
        '''
        Convert 3D spline into discrete 3D points

        Points are sampled either with a constant sampling rate or at the given timestamps t

        Outputs are 3D points
        '''
        
        tck, interval = self.spline['tck'], self.spline['int']
        self.traj = np.empty([4,0])

        if t is not None:
            assert len(t.shape)==1, 'Input timestamps must be a 1D array'
            timestamp = t
        else:
            timestamp = np.arange(interval[0,0], interval[1,-1], sampling_rate)

        for i in range(interval.shape[1]):
            t_part = timestamp[np.logical_and(timestamp>=interval[0,i], timestamp<=interval[1,i])]
            try:
                traj_part = np.asarray(interpolate.splev(t_part, tck[i]))
            except:
                continue
            self.traj = np.hstack((self.traj, np.vstack((t_part,traj_part))))

        assert (self.traj[0,1:] > self.traj[0,:-1]).all()

        return self.traj


    def error_cam(self,cam_id,mode='dist'):
        '''
        Calculate the reprojection errors for a given camera

        Different modes are available: 'dist', 'xy_1D', 'xy_2D', 'each'
        '''

        tck, interval = self.spline['tck'], self.spline['int']
        self.detection_to_global(cam_id)

        _, idx = self.sampling(self.detections_global[cam_id], interval, belong=True)
        detect = np.empty([3,0])
        point_3D = np.empty([3,0])
        for i in range(interval.shape[1]):
            detect_part = self.detections_global[cam_id][:,idx==i+1]
            if detect_part.size:
                detect = np.hstack((detect,detect_part))
                point_3D = np.hstack((point_3D, np.asarray(interpolate.splev(detect_part[0], tck[i]))))

        X = util.homogeneous(point_3D)
        x = detect[1:]
        x_cal = self.cameras[cam_id].projectPoint(X)

        if mode == 'dist':
            return ep.reprojection_error(x, x_cal)
        elif mode == 'xy_1D':
            return np.concatenate((abs(x_cal[0]-x[0]),abs(x_cal[1]-x[1])))
        elif mode == 'xy_2D':
            return np.vstack((abs(x_cal[0]-x[0]),abs(x_cal[1]-x[1])))
        elif mode == 'each':
            error_x = np.zeros_like(self.detections[cam_id][0])
            error_y = np.zeros_like(self.detections[cam_id][0])
            error_x[idx.astype(bool)] = abs(x_cal[0]-x[0])
            error_y[idx.astype(bool)] = abs(x_cal[1]-x[1])
            return np.concatenate((error_x, error_y))

    
    def compute_visibility(self):
        '''
        Decide for each raw detection if it is visible from current 3D spline
        '''

        self.visible = []
        interval = self.spline['int']
        self.detection_to_global()

        for cam_id in range(self.numCam):
            _, visible = self.sampling(self.detections_global[cam_id], interval, belong=True)
            self.visible.append(visible)


    def BA(self, numCam, max_iter=10):
        '''
        Bundle Adjustment with multiple splines

        The camera order is assumed to be the same as self.sequence
        '''

        def error_BA(x):
            '''
            Input is the model (parameters that need to be optimized)
            '''

            # Assign parameters to the class attributes
            sections = [numCam, numCam*2, numCam*2+numCam*self.cam_model]
            model_parts = np.split(x, sections)
            self.alpha[self.sequence[:numCam]], self.beta[self.sequence[:numCam]] = model_parts[0], model_parts[1]

            cams = np.split(model_parts[2],numCam)
            for i in range(numCam):
                self.cameras[self.sequence[i]].vector2P(cams[i], n=self.cam_model) 
            
            spline_parts = np.split(model_parts[3],idx_spline[0,1:])
            for i in range(len(spline_parts)):
                spline_i = spline_parts[i].reshape(3,-1)
                self.spline['tck'][i][1] = [spline_i[0],spline_i[1],spline_i[2]]
            
            # Compute errors
            error = np.array([])
            for i in range(numCam):
                error_each = self.error_cam(self.sequence[i], mode='each')
                error = np.concatenate((error, error_each))

            return error


        def jac_BA(near=3):

            num_param = len(model)
            self.compute_visibility()

            jac = np.empty([0,num_param])
            for i in range(numCam):
                cam_id = self.sequence[i]
                num_detect = self.detections[cam_id].shape[1]

                # consider only reprojection in x direction, which is the same in y direction
                jac_cam = np.zeros((num_detect, num_param))

                # alpha and beta
                jac_cam[:,[i,i+numCam]] = 1

                # camera parameters
                start = 2*numCam+i*self.cam_model
                jac_cam[:,start:start+self.cam_model] = 1

                # spline parameters
                for j in range(num_detect):
                    spline_id = self.visible[cam_id][j]

                    # Find the corresponding spline for each detecion
                    if spline_id:
                        spline_id -= 1
                        knot = self.spline['tck'][spline_id][0][2:-2]
                        timestamp = self.detections_global[cam_id][0,j]
                        knot_idx = np.argsort(abs(knot-timestamp))[:near]
                        knot_idx = np.concatenate((knot_idx, knot_idx+len(knot), knot_idx+2*len(knot)))
                        jac_cam[j,idx_spline_sum[0,spline_id]+knot_idx] = 1
                    else:
                        jac_cam[j,:] = 0

                jac = np.vstack((jac, np.tile(jac_cam,(2,1))))

            # fix the first camera
            # jac[:,[0,numCam]], jac[:,2*numCam+4:2*numCam+10] = 0, 0

            return jac


        starttime = datetime.now()
        
        '''Before BA'''
        # Define Parameters that will be optimized
        model_alpha = self.alpha[self.sequence[:numCam]]
        model_beta = self.beta[self.sequence[:numCam]]

        model_cam = np.array([])
        for i in self.sequence[:numCam]:
            model_cam = np.concatenate((model_cam, self.cameras[i].P2vector(n=self.cam_model)))

        # Reorganized splines into 1D and record indices of each spline
        num_spline = len(self.spline['tck'])
        idx_spline = np.zeros((2,num_spline),dtype=int)
        start = 0
        model_spline = np.array([])
        for i in range(num_spline):
            model_spline_i = np.ravel(self.spline['tck'][i][1])
            model_spline = np.concatenate((model_spline, model_spline_i))

            end = start + len(model_spline_i)
            idx_spline[:,i] = [start,end]
            start = end

        model_other = np.concatenate((model_alpha, model_beta, model_cam))
        idx_spline_sum = idx_spline + len(model_other)
        model = np.concatenate((model_other, model_spline))
        assert idx_spline_sum[-1,-1] == len(model), 'Wrong with spline indices'

        # Set the Jacobian matrix
        print('\nComputing the sparse structure of the Jacobian matrix...\n')
        A = jac_BA()

        '''Compute BA'''
        print('Doing BA with {} cameras...\n'.format(numCam))
        fn = lambda x: error_BA(x)
        res = least_squares(fn,model,jac_sparsity=A,tr_solver='lsmr',max_nfev=max_iter)

        '''After BA'''
        # Assign the optimized model to alpha, beta, cam, and spline
        sections = [numCam, numCam*2, numCam*2+numCam*self.cam_model]
        model_parts = np.split(res.x, sections)
        self.alpha[self.sequence[:numCam]], self.beta[self.sequence[:numCam]] = model_parts[0], model_parts[1]

        cams = np.split(model_parts[2],numCam)
        for i in range(numCam):
            self.cameras[self.sequence[i]].vector2P(cams[i], n=self.cam_model) 
        
        spline_parts = np.split(model_parts[3],idx_spline[0,1:])
        for i in range(len(spline_parts)):
            spline_i = spline_parts[i].reshape(3,-1)
            self.spline['tck'][i][1] = [spline_i[0],spline_i[1],spline_i[2]]

        # Update global timestamps for each serie of detections
        self.detection_to_global()

        print('BA finished in: {}'.format(datetime.now()-starttime))

        return res


    def remove_outliers(self, cams, thres=30):
        '''
        Not done yet!
        '''

        for i in cams:
            error_all = self.error_cam(i,mode='each')
            error_xy = np.split(error_all,2)
            error = np.sqrt(error_xy[0]**2 + error_xy[1]**2)

            self.detections[i] = self.detections[i][:,error<thres]
            self.detection_to_global(i)


    def get_camera_pose(self, cam_id, error=8, verbose=0):
        '''
        Get the absolute pose of a camera by solving the PnP problem.

        Take care with DISTORSION model!
        '''
        
        tck, interval = self.spline['tck'], self.spline['int']
        self.detection_to_global(cam_id)

        _, idx = self.sampling(self.detections_global[cam_id], interval, belong=True)
        detect = np.empty([3,0])
        point_3D = np.empty([3,0])
        for i in range(interval.shape[1]):
            detect_part = self.detections_global[cam_id][:,idx==i+1]
            if detect_part.size:
                detect = np.hstack((detect,detect_part))
                point_3D = np.hstack((point_3D, np.asarray(interpolate.splev(detect_part[0], tck[i]))))

        # PnP solution from OpenCV
        N = point_3D.shape[1]
        objectPoints = np.ascontiguousarray(point_3D.T).reshape((N,1,3))
        imagePoints  = np.ascontiguousarray(detect[1:].T).reshape((N,1,2))
        cam_dist = self.cameras[cam_id].d
        if len(cam_dist)==2:
            distCoeffs = np.append(cam_dist, [0, 0, 0])
        elif len(cam_dist)==3:
            distCoeffs = np.array([cam_dist[0], cam_dist[1], 0, 0, cam_dist[2]])
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, self.cameras[cam_id].K, distCoeffs, reprojectionError=error)

        self.cameras[cam_id].R = cv2.Rodrigues(rvec)[0]
        self.cameras[cam_id].t = tvec.reshape(-1,)
        self.cameras[cam_id].compose()

        if verbose:
            print('{} out of {} points are inliers for PnP'.format(inliers.shape[0], N))
            

    def triangulate(self, cam_id, cams, factor_t2s=0.001, factor_s2t=0.02, thres=0, refit=True, verbose=0):
        '''
        Triangulate new points to the existing 3D spline and optionally refit it

        cam_id is the new camera
        
        cams must be an iterable that contains cameras that have been processed to build the 3D spline
        '''

        assert self.cameras[cam_id].P is not None, 'The camera pose must be computed first'
        tck, interval = self.spline['tck'], self.spline['int']
        self.detection_to_global(cam_id)

        # Find detections from this camera that haven't been triangulated yet
        _, idx_ex = self.sampling(self.detections_global[cam_id], interval)
        detect_new = self.detections_global[cam_id][:, np.logical_not(idx_ex)]

        # Matching these detections with detections from previous cameras and triangulate them
        X_new = np.empty([4,0])
        for i in cams:
            self.detection_to_global(i)
            detect_ex = self.detections_global[i]

            # Detections of previous cameras are interpolated, no matter the fps
            try:
                x1, x2 = self.match_overlap(detect_new, detect_ex)
            except:
                continue
            else:
                P1, P2 = self.cameras[cam_id].P, self.cameras[i].P
                X_i = ep.triangulate_matlab(x1[1:], x2[1:], P1, P2)
                X_i = np.vstack((x1[0], X_i[:-1]))

                # Check reprojection error directly after triangulation, preserve those with small error
                if thres:
                    err_1 = ep.reprojection_error(x1[1:], self.cameras[cam_id].projectPoint(X_i[1:]))
                    err_2 = ep.reprojection_error(x2[1:], self.cameras[i].projectPoint(X_i[1:]))
                    mask = np.logical_and(err_1<thres, err_2<thres)
                    X_i = X_i[:, mask]
                    
                    if verbose:
                        print('{} out of {} points are triangulated'.format(sum(mask), len(err_1)))

                X_new = np.hstack((X_new, X_i))

                if verbose:
                    print('{} points are triangulated into the 3D spline'.format(X_i.shape[1]))

        _, idx_empty = self.sampling(X_new, interval)
        assert sum(idx_empty)==0, 'Points should not be triangulated into the existing part of the 3D spline'

        # Add these points to the discrete 3D trajectory
        self.spline_to_traj(sampling_rate=factor_s2t)
        self.traj = np.hstack((self.traj, X_new))
        _, idx = np.unique(self.traj[0], return_index=True)
        self.traj = self.traj[:, idx]

        # refit the 3D spline if wanted
        if refit:
            self.traj_to_spline(smooth_factor=factor_t2s)

        return X_new
        

    def plot_reprojection(self,interval,match=True):
        '''
        Given temporal sections of the trajectory, plot the 2D reprojection of these sections for
        each possible camera
        '''

        assert interval.shape[0]==2

        for i in range(self.numCam):
            detect_i, _ = self.sampling(self.detections_global[i],interval)
            traj = self.spline_to_traj(t=detect_i[0])
            
            if traj.size:
                if match:
                    xy,x_ind,y_ind = np.intersect1d(detect_i[0],traj[0],assume_unique=True,return_indices=True)
                    detect_i = detect_i[:,x_ind]
                    traj = traj[:,y_ind]

                try:
                    repro = self.cameras[i].projectPoint(traj[1:])
                except:
                    continue
                
                plt.figure(figsize=(12, 10))
                plt.scatter(detect_i[1],detect_i[2],c='red')
                plt.scatter(repro[0],repro[1],c='blue')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.suptitle('Camera {}: undistorted detection (red) vs reporjection (blue)'.format(i))

        plt.show()


class Scene_single_spline(Scene):
    '''
    Should NOT be used anymore, only a back up possibly with bugs. Apply the multi-spline class instead if possible.
    '''

    def __init__(self):
        super().__init__()
        self.alpha = None
        self.detections_global = []


    def init_alpha(self,prior=[]):
        '''Initialize alpha for each camera based on the ratio of fps'''

        if len(prior):
            assert len(prior) == self.numCam, 'Number of input must be the same as the number of cameras'
            self.alpha = prior
        else:
            self.alpha = np.ones(self.numCam)
            fps_ref = self.cameras[self.sequence[0]].fps
            for i in range(self.numCam):
                self.alpha[i] = fps_ref / self.cameras[i].fps


    def detection_to_global(self, *cam, distortion=True):
        '''
        Convert frame indices of raw detections into the global timeline.

        DISTORTION is considered.

        Input is an iterable that specifies which detection(s) to compute.

        If no input, all detections will be converted.
        '''

        assert len(self.alpha)==self.numCam and len(self.beta)==self.numCam, 'The Number of alpha and beta is wrong'

        if not distortion:
            if len(cam):
                for i in cam:
                    detect_global = np.vstack((self.alpha[i] * self.detections[i][0] + self.beta[i], self.detections[i][1:]))
                    self.detections_global[i] = detect_global
            else:
                self.detections_global = []
                for i in range(self.numCam):
                    detect_global = np.vstack((self.alpha[i] * self.detections[i][0] + self.beta[i], self.detections[i][1:]))
                    self.detections_global.append(detect_global)
        else:
            if len(cam):
                for i in cam:
                    detect_temp = detect_undistort([self.detections[i]], [self.cameras[i]])
                    self.detections_undist[i] = detect_temp[0]
                    detect_global = np.vstack((self.alpha[i] * self.detections_undist[i][0] + self.beta[i], self.detections_undist[i][1:]))
                    self.detections_global[i] = detect_global
            else:
                self.detections_undist = detect_undistort(self.detections, self.cameras)
                self.detections_global = []
                for i in range(self.numCam):
                    detect_global = np.vstack((self.alpha[i] * self.detections_undist[i][0] + self.beta[i], self.detections_undist[i][1:]))
                    self.detections_global.append(detect_global)


    def cut_detection(self,second=1):
        '''
        Truncate the starting and end part of each continuous part of the detections
        '''

        for i in range(self.numCam):
            detect = self.detections[i]
            interval = self.find_intervals(detect[0])
            cut = int(self.cameras[i].fps * second)

            interval_long = interval[:,interval[1]-interval[0]>cut*2]
            interval_long[0] += cut
            interval_long[1] -= cut

            assert (interval_long[1]-interval_long[0]>=0).all()

            self.detections[i], _ = self.sampling(detect,interval_long)


    def find_intervals(self,x,gap=5):
        '''
        Given indices of detections, return a matrix that contains the start and the end of each
        continues part.
        
        Input indices must be in ascending order. 
        
        The gap defines the maximal interruption, with which it's still considered as continues. 
        '''

        assert len(x.shape)==1, 'Input must be a 1D-array'

        interval = np.array([[x[0]],[np.nan]])
        for i in range(1,len(x)):
            assert x[i]-x[i-1]>0, 'Input must be in ascending order'
            if x[i]-x[i-1] > gap:
                interval[1,-1] = x[i-1]
                interval = np.append(interval,[[x[i]],[np.nan]],axis=1)
        interval[1,-1] = x[-1]
        return interval


    def sampling(self,x,interval):
        '''
        Sample points from the input which are inside the given intervals
        '''

        if len(x.shape)==1:
            timestamp = x
        elif len(x.shape)==2:
            assert x.shape[0]==3 or x.shape[0]==4, 'Input should be 1D array or 2D array with 3 or 4 rows'
            timestamp = x[0]
  
        t_bool = np.zeros_like(timestamp, dtype=bool)
        for i in interval.T:
            t_bool = t_bool | np.logical_xor(timestamp-i[0] > 0, timestamp-i[1] > 0)
        idx_ts = np.where(t_bool)[0]

        if len(x.shape)==1:
            return x[idx_ts], idx_ts
        elif len(x.shape)==2:
            return x[:,idx_ts], idx_ts


    def match_overlap(self,x,y):
        '''
        Given two inputs in the same timeline (global), return the parts of them which are temporally overlapped

        Important: it's assumed that x has a higher frequency (fps) so that points are interpolated in y
        '''

        interval = self.find_intervals(y[0])
        x_s, _ = self.sampling(x, interval)

        tck, u = interpolate.splprep(y[1:],u=y[0],s=0,k=3)
        y_s = np.asarray(interpolate.splev(x_s[0],tck))
        y_s = np.vstack((x_s[0],y_s))

        assert (x_s[0] == y_s[0]).all(), 'Both outputs should have the same timestamps'

        return x_s, y_s


    def init_traj(self,error=10,inlier_only=False):
        '''
        Select the first two cams in the sequence, compute fundamental matrix, triangulate points
        '''

        t1, t2 = self.sequence[0], self.sequence[1]
        K1, K2 = self.cameras[t1].K, self.cameras[t2].K

        # Find correspondences
        if self.cameras[t1].fps > self.cameras[t2].fps:
            d1, d2 = self.match_overlap(self.detections_global[t1], self.detections_global[t2])
        else:
            d2, d1 = self.match_overlap(self.detections_global[t2], self.detections_global[t1])
        
        # Compute fundamental matrix
        F,inlier = ep.computeFundamentalMat(d1[1:],d2[1:],error=error)
        E = np.dot(np.dot(K2.T,F),K1)

        if not inlier_only:
            inlier = np.ones(len(inlier))
        x1, x2 = util.homogeneous(d1[1:,inlier==1]), util.homogeneous(d2[1:,inlier==1])

        # Find corrected corresponding points for optimal triangulation
        N = d1[1:,inlier==1].shape[1]
        pts1=d1[1:,inlier==1].T.reshape(1,-1,2)
        pts2=d2[1:,inlier==1].T.reshape(1,-1,2)
        m1,m2 = cv2.correctMatches(F,pts1,pts2)
        x1,x2 = util.homogeneous(np.reshape(m1,(-1,2)).T), util.homogeneous(np.reshape(m2,(-1,2)).T)

        # Triangulte points
        X, P = ep.triangulate_from_E(E,K1,K2,x1,x2)
        self.traj = np.vstack((d1[0][inlier==1],X[:-1]))

        # Assign the camera matrix for these two cameras
        self.cameras[t1].P = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
        self.cameras[t2].P = np.dot(K2,P)
        self.cameras[t1].decompose()
        self.cameras[t2].decompose()


    def traj_to_spline(self,smooth_factor=0.01):
        '''
        Convert discrete 3D trajectory into spline representation

        Outputs are spline parameters and intervals
        '''

        interval = self.find_intervals(self.traj[0])
        tck, u = interpolate.splprep(self.traj[1:],u=self.traj[0],s=smooth_factor,k=3)
        self.spline = [tck, interval]

        return self.spline


    def spline_to_traj(self,sampling_rate=1,t=None):
        '''
        Convert 3D spline into discrete 3D points

        Points are sampled either with a constant sampling rate or at the given timestamps t

        Outputs are 3D points
        '''
        
        tck, interval = self.spline[0], self.spline[1]

        if t is not None:
            assert len(t.shape)==1, 'Input timestamps must be a 1D array'
            timestamp = t
        else:
            timestamp = np.array([])
            for i in range(interval.shape[1]):
                timestamp = np.concatenate((timestamp, np.arange(interval[0,i], interval[1,i], sampling_rate)))

        traj = np.asarray(interpolate.splev(timestamp, tck))
        self.traj = np.vstack((timestamp,traj))

        return self.traj


    def error_cam(self,cam_id,mode='dist'):
        '''
        Calculate the reprojection errors for a given camera

        Different modes are available: 'dist', 'xy_1D', 'xy_2D', 'each'
        '''

        tck, interval = self.spline[0], self.spline[1]
        self.detection_to_global(cam_id)

        detect_2D, idx = self.sampling(self.detections_global[cam_id], interval)
        point_3D = np.asarray(interpolate.splev(detect_2D[0], tck))

        X = util.homogeneous(point_3D)
        x = detect_2D[1:]
        x_cal = self.cameras[cam_id].projectPoint(X)

        if mode == 'dist':
            return ep.reprojection_error(x, x_cal)
        elif mode == 'xy_1D':
            return np.concatenate((abs(x_cal[0]-x[0]),abs(x_cal[1]-x[1])))
        elif mode == 'xy_2D':
            return np.vstack((abs(x_cal[0]-x[0]),abs(x_cal[1]-x[1])))
        elif mode == 'each':
            error_x = np.zeros_like(self.detections[cam_id][0])
            error_y = np.zeros_like(self.detections[cam_id][0])
            error_x[idx] = abs(x_cal[0]-x[0])
            error_y[idx] = abs(x_cal[1]-x[1])
            return np.concatenate((error_x, error_y))

    
    def compute_visibility(self):
        '''
        Decide for each raw detection if it is visible from current 3D spline
        '''

        self.visible = []
        interval = self.spline[1]
        self.detection_to_global()

        for i in range(self.numCam):
            visible = np.zeros_like(self.detections[i][0])
            _, idx = self.sampling(self.detections_global[i], interval)
            visible[idx] = 1
            self.visible.append(visible)


    def BA(self, numCam, max_iter=10):
        '''
        '''

        def error_BA(x):
            '''
            Input is the model (parameters that need to be optimized)
            '''

            # Assign parameters to the class attributes
            sections = [numCam, numCam*2, numCam*2+numCam*12]
            p = np.split(x, sections)
            self.alpha[self.sequence[:numCam]], self.beta[self.sequence[:numCam]] = p[0], p[1]

            cams = np.split(p[2],numCam)
            for i in range(numCam):
                self.cameras[self.sequence[i]].vector2P(cams[i], n=12) 
            
            spline_temp = p[3].reshape(3,-1)
            self.spline[0][1][0], self.spline[0][1][1], self.spline[0][1][2] = spline_temp[0], spline_temp[1], spline_temp[2]

            # Compute errors
            error = np.array([])
            for i in range(numCam):
                error_each = self.error_cam(self.sequence[i], mode='each')
                error = np.concatenate((error, error_each))

            return error


        def jac_BA(near=3):
            '''
            Inputs are the model (parameters that need to be optimized) and the knot vector of 3D spline
            '''

            knot = self.spline[0][0][2:-2]
            assert len(knot)==len(self.spline[0][1][0]), 'The Number of knots is wrong'

            num_param = len(model)
            self.compute_visibility()

            jac = np.zeros_like(model)
            for i in range(numCam):
                cam_id = self.sequence[i]
                num_detect = self.detections[cam_id].shape[1]

                # consider only reprojection in x direction, which is the same in y direction
                jac_cam = np.zeros((num_detect, num_param))

                # alpha and beta
                jac_cam[:,[i,i+numCam]] = 1

                # camera parameters
                start = 2*numCam+i*12
                jac_cam[:,start:start+12] = 1

                # spline parameters
                for j in range(num_detect):
                    if self.visible[cam_id][j]:
                        timestamp = self.detections_global[cam_id][0,j]
                        knot_idx = np.argsort(abs(knot-timestamp))[:near]
                        knot_idx = np.concatenate((knot_idx, knot_idx+len(knot), knot_idx+2*len(knot)))
                        jac_cam[j,2*numCam+12*numCam+knot_idx] = 1
                    else:
                        jac_cam[j,:] = 0

                jac = np.vstack((jac, np.tile(jac_cam,(2,1))))

            # fix the first camera
            jac[:,[0,numCam]], jac[:,2*numCam+4:2*numCam+10] = 0, 0

            return jac[1:]


        start = datetime.now()
        
        # Define Parameters that will be optimized
        model_alpha = self.alpha[self.sequence[:numCam]]
        model_beta = self.beta[self.sequence[:numCam]]

        model_cam = np.array([])
        for i in self.sequence[:numCam]:
            model_cam = np.concatenate((model_cam, self.cameras[i].P2vector(n=12)))

        model_spline = np.ravel(self.spline[0][1])
        model = np.concatenate((model_alpha, model_beta, model_cam, model_spline))

        # Set the Jacobian matrix
        print('\nComputing the sparse structure of the Jacobian matrix...\n')
        A = jac_BA()

        # Compute non-linear optimization
        print('Doing BA with {} cameras...\n'.format(numCam))
        fn = lambda x: error_BA(x)
        res = least_squares(fn,model,jac_sparsity=A,tr_solver='lsmr',max_nfev=max_iter)

        # Assign the optimized model to the scene
        sections = [numCam, numCam*2, numCam*2+numCam*12]
        p = np.split(res.x, sections)
        self.alpha[self.sequence[:numCam]], self.beta[self.sequence[:numCam]] = p[0], p[1]

        cams = np.split(p[2],numCam)
        for i in range(numCam):
            self.cameras[self.sequence[i]].vector2P(cams[i], n=12) 
        
        spline_temp = p[3].reshape(3,-1)
        self.spline[0][1][0], self.spline[0][1][1], self.spline[0][1][2] = spline_temp[0], spline_temp[1], spline_temp[2]

        # Update global timestamps for each serie of detections
        self.detection_to_global()

        print('BA finished in: {}'.format(datetime.now()-start))

        return res


    def remove_outliers(self, *cams, thres=30):
        '''
        Not done yet!
        '''

        for i in cams:
            error_all = self.error_cam(i,mode='each')
            error_xy = np.split(error_all,2)
            error = np.sqrt(error_xy[0]**2 + error_xy[1]**2)


    def get_camera_pose(self, cam_id, error=8, verbose=0):
        '''
        Get the absolute of a camera by solving the PnP problem.

        Take care with DISTORSION model!
        '''
        
        tck, interval = self.spline[0], self.spline[1]
        self.detection_to_global(cam_id)

        detect_2D, idx = self.sampling(self.detections_global[cam_id], interval)
        point_3D = np.asarray(interpolate.splev(detect_2D[0], tck))

        X = util.homogeneous(point_3D)
        x = util.homogeneous(detect_2D[1:])

        # PnP solution from OpenCV
        N = X.shape[1]
        objectPoints = np.ascontiguousarray(X[:3].T).reshape((N,1,3))
        imagePoints  = np.ascontiguousarray(x[:2].T).reshape((N,1,2))
        distCoeffs = np.append(self.cameras[cam_id].d, [0, 0, 0])
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, self.cameras[cam_id].K, distCoeffs, reprojectionError=error)

        self.cameras[cam_id].R = cv2.Rodrigues(rvec)[0]
        self.cameras[cam_id].t = tvec.reshape(-1,)
        self.cameras[cam_id].compose()

        if verbose:
            print('{} out of {} points are inliers for PnP'.format(inliers.shape[0], X.shape[1]))
            

    def triangulate(self, cam_id, cams, factor_t2s=0.001, factor_s2t=0.02, refit=True, verbose=0):
        '''
        Triangulate new points to the existing 3D spline and optionally refit it

        cam_id is the new camera
        
        cams must be an iterable that contains cameras that have been processed to build the 3D spline
        '''

        assert self.cameras[cam_id].P is not None, 'The camera pose must be computed first'
        tck, interval = self.spline[0], self.spline[1]
        self.detection_to_global(cam_id)

        # Find detections from this camera that haven't been triangulated yet
        _, idx_ex = self.sampling(self.detections_global[cam_id], interval)
        num_detect = self.detections_global[cam_id].shape[1]
        idx_new = np.setdiff1d(np.arange(num_detect), idx_ex)
        detect_new = self.detections_global[cam_id][:, idx_new]

        # Matching these detections with detections from previous cameras and triangulate them
        X_new = np.empty([4,0])
        for i in cams:
            self.detection_to_global(i)
            detect_ex = self.detections_global[i]

            # Detections of previous cameras are interpolated, no matter the fps
            try:
                x1, x2 = self.match_overlap(detect_new, detect_ex)
            except:
                continue
            else:
                P1, P2 = self.cameras[cam_id].P, self.cameras[i].P
                X_i = ep.triangulate_matlab(x1[1:], x2[1:], P1, P2)
                X_i = np.vstack((x1[0], X_i[:-1]))
                X_new = np.hstack((X_new, X_i))

                if verbose:
                    print('{} points are triangulated into the 3D spline'.format(X_i.shape[1]))

        _, idx_empty = self.sampling(X_new, interval)
        assert len(idx_empty)==0, 'Points should not be triangulated into the existing part of the 3D spline'

        # Add these points to the discrete 3D trajectory
        self.spline_to_traj(sampling_rate=factor_s2t)
        self.traj = np.hstack((self.traj, X_new))
        _, idx = np.unique(self.traj[0], return_index=True)
        self.traj = self.traj[:, idx]

        # refit the 3D spline if wanted
        if refit:
            self.traj_to_spline(smooth_factor=factor_t2s)

        return X_new

        
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
        self.fps = kwargs.get('fps')

        if kwargs.get('tan') is None:
            self.tan = np.array([0,0],dtype=float)
        else:
            self.tan = kwargs.get('tan')

    def projectPoint(self,X):

        assert self.P is not None, 'The projection matrix P has not been calculated yet'
        if X.shape[0] == 3:
            X = util.homogeneous(X)
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
            return np.concatenate((k,r,self.t,self.d[:2]))
        elif n==13:
            k = np.array([self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]])
            return np.concatenate((k,r,self.t,self.d[:3]))


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
            self.d = vector[10:12]
        elif n==13:
            self.K = np.diag((1,1,1)).astype(float)
            self.K[0,0], self.K[1,1] = vector[0], vector[1]
            self.K[:2,-1] = vector[2:4]
            self.R = cv2.Rodrigues(vector[4:7])[0]
            self.t = vector[7:10]
            self.d = vector[10:13]

        self.compose()
        return self.P
    

    def undist_point(self,points):
        
        assert points.shape[0]==2, 'Input must be a 2D array'

        num = points.shape[1]

        if len(self.d)==2:
            distCoeff = np.concatenate((self.d,self.tan))
        elif len(self.d)==3:
            if self.d[-1] == 0:
                distCoeff = np.concatenate((self.d,self.tan))
            else:
                distCoeff = np.array([self.d[0], self.d[1], self.tan[0], self.tan[1], self.d[2]])
        else:
            raise Exception('Wrong distortion coefficients')

        src = np.ascontiguousarray(points.T).reshape((num,1,2))
        dst = cv2.undistortPoints(src, self.K, distCoeff)
        dst_unnorm = np.dot(self.K, util.homogeneous(dst.reshape((num,2)).T))

        return dst_unnorm[:2]


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

    return detections


def detect_spline_fitting(d,smooth):
    detections = copy.deepcopy(d)
    for i in range(len(detections)):
        detections[i][1] = util.spline_fitting(detections[i][1],detections[i][0],detections[i][0],k=3,s=smooth[i])
        detections[i][2] = util.spline_fitting(detections[i][2],detections[i][0],detections[i][0],k=3,s=smooth[i])
    return detections


def detect_to_track(detections,beta):

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
    n_cam = 12
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