# Classes that are common to the entire project
import numpy as np
from tools import util
import reconstruction.epipolar as ep
import cv2
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
        self.numCam = 0
        self.cameras = []
        self.cam_model = 12
        self.cam_size = []
        self.detections = []
        self.detections_raw = []
        self.detections_global = []
        self.alpha = []
        self.beta = []
        self.traj = []
        self.sequence = []
        self.visible = []
        self.setting = None
        self.gps = None
        self.spline = {'tck':[], 'int':[]}
        self.rs = []
        

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
                    detect_global = np.vstack((self.alpha[i] * self.detections[i][0] + self.beta[i] + self.rs[i] * self.detections[i][2] / self.cam_size[i][1], self.detections[i][1:]))
                    self.detections_global[i] = detect_global
            else:
                self.detections_global = []
                for i in range(self.numCam):
                    detect_global = np.vstack((self.alpha[i] * self.detections[i][0] + self.beta[i] + self.rs[i] * self.detections[i][2] / self.cam_size[i][1], self.detections[i][1:]))
                    self.detections_global.append(detect_global)
        else:
            if len(cam):
                for i in cam:
                    timestamp = self.alpha[i] * self.detections[i][0] + self.beta[i] + self.rs[i] * self.detections[i][2] / self.cam_size[i][1]
                    detect = self.cameras[i].undist_point(self.detections[i][1:])
                    self.detections_global[i] = np.vstack((timestamp,detect))
            else:
                self.detections_global = []
                for i in range(self.numCam):
                    timestamp = self.alpha[i] * self.detections[i][0] + self.beta[i] + self.rs[i] * self.detections[i][2] / self.cam_size[i][1]
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


    def BA(self, numCam, max_iter=10, rs=False):
        '''
        Bundle Adjustment with multiple splines

        The camera order is assumed to be the same as self.sequence
        '''

        def error_BA(x):
            '''
            Input is the model (parameters that need to be optimized)
            '''

            # Assign parameters to the class attributes
            sections = [numCam, numCam*2, numCam*3, numCam*3+numCam*self.cam_model]
            model_parts = np.split(x, sections)
            self.alpha[self.sequence[:numCam]], self.beta[self.sequence[:numCam]], self.rs[self.sequence[:numCam]] = model_parts[0], model_parts[1], model_parts[2]

            cams = np.split(model_parts[3],numCam)
            for i in range(numCam):
                self.cameras[self.sequence[i]].vector2P(cams[i], n=self.cam_model) 
            
            spline_parts = np.split(model_parts[4],idx_spline[0,1:])
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

                # rolling shutter
                if rs:
                    jac_cam[:,i+numCam*2] = 1
                else:
                    jac_cam[:,i+numCam*2] = 0

                # camera parameters
                start = 3*numCam+i*self.cam_model
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
        model_rs = self.rs[self.sequence[:numCam]]

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

        model_other = np.concatenate((model_alpha, model_beta, model_rs, model_cam))
        idx_spline_sum = idx_spline + len(model_other)
        model = np.concatenate((model_other, model_spline))
        assert idx_spline_sum[-1,-1] == len(model), 'Wrong with spline indices'

        # Set the Jacobian matrix
        A = jac_BA()

        '''Compute BA'''
        print('Doing BA with {} cameras...\n'.format(numCam))
        fn = lambda x: error_BA(x)
        res = least_squares(fn,model,jac_sparsity=A,tr_solver='lsmr',max_nfev=max_iter)

        '''After BA'''
        # Assign the optimized model to alpha, beta, cam, and spline
        sections = [numCam, numCam*2, numCam*3, numCam*3+numCam*self.cam_model]
        model_parts = np.split(res.x, sections)
        self.alpha[self.sequence[:numCam]], self.beta[self.sequence[:numCam]], self.rs[self.sequence[:numCam]] = model_parts[0], model_parts[1], model_parts[2]

        cams = np.split(model_parts[3],numCam)
        for i in range(numCam):
            self.cameras[self.sequence[i]].vector2P(cams[i], n=self.cam_model) 
        
        spline_parts = np.split(model_parts[4],idx_spline[0,1:])
        for i in range(len(spline_parts)):
            spline_i = spline_parts[i].reshape(3,-1)
            self.spline['tck'][i][1] = [spline_i[0],spline_i[1],spline_i[2]]

        # Update global timestamps for each serie of detections
        self.detection_to_global()

        return res


    def remove_outliers(self, cams, thres=30):
        '''
        Not done yet!
        '''

        if thres:
            for i in cams:
                error_all = self.error_cam(i,mode='each')
                error_xy = np.split(error_all,2)
                error = np.sqrt(error_xy[0]**2 + error_xy[1]**2)

                self.detections[i] = self.detections[i][:,error<thres]
                self.detection_to_global(i)

                print('{} out of {} detections are removed for camera {}'.format(sum(error>=thres),sum(error!=0),i))


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
        

    def plot_reprojection(self,interval=np.array([[-np.inf],[np.inf]]),match=True):
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
