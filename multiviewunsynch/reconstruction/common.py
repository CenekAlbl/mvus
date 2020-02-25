# Classes that are common to the entire project
import numpy as np
from tools import util
import cv2
import json
from reconstruction import epipolar as ep
from reconstruction import synchronization as sync
from datetime import datetime
from scipy.optimize import least_squares
from scipy import interpolate
from scipy.sparse import lil_matrix, vstack
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tools import visualization as vis
from tools import util 


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
        self.detections = []
        self.detections_raw = []
        self.detections_global = []
        self.alpha = []
        self.beta = []
        self.cf = []
        self.traj = []
        self.traj_len = []
        self.sequence = []
        self.visible = []
        self.settings = []
        self.gt = []
        self.out = {}
        self.spline = {'tck':[], 'int':[]}
        self.rs = []
        self.ref_cam = 0
        self.find_order = True
        

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


    def init_alpha(self,*prior):
        '''Initialize alpha for each camera based on the ratio of fps'''

        if len(prior):
            assert len(prior) == self.numCam, 'Number of input must be the same as the number of cameras'
            self.alpha = prior
        else:
            self.alpha = np.ones(self.numCam)
            fps_ref = self.cameras[self.ref_cam].fps
            for i in range(self.numCam):
                self.alpha[i] = fps_ref / self.cameras[i].fps


    def detection_to_global(self,*cam,motion_prior=False):
        '''
        Convert frame indices of raw detections into the global timeline.

        Input is an iterable that specifies which detection(s) to compute.

        If no input, all detections will be converted.
        '''

        assert len(self.alpha)==self.numCam and len(self.beta)==self.numCam, 'The Number of alpha and beta is wrong'

        if len(cam):
            cams = cam
            if type(cams[0]) != int:
                cams = cams[0]
        else:
            cams = range(self.numCam)
            self.detections_global = [[] for i in cams]

        for i in cams:
            timestamp = self.alpha[i] * (self.detections[i][0] + self.rs[i] * self.detections[i][2] / self.cameras[i].resolution[1]) + self.beta[i] 
            detect = self.cameras[i].undist_point(self.detections[i][1:]) if self.settings['undist_points'] else self.detections[i][1:]
            self.detections_global[i] = np.vstack((timestamp, detect))

            if motion_prior:
                if (self.global_traj[1] == i).any():
                    # Update glob_traj timestamps for current camera
                    # np.logical_and(timestamp>=interval[0,i], timestamp<=interval[1,i]) 
                    #temp_glob_traj = self.global_traj[:,np.logical_and(self.global_traj[1] == i,self.global_traj[3] in ]
                    temp_glob_traj = self.global_traj[:,self.global_traj[1] == i]
                    # Save traj. point locations in global_traj before update
                    #temp_glob_traj_mask = np.isin(self.global_traj[3],temp_glob_traj[3])
                    temp_glob_traj_mask = np.where(self.global_traj[1] == i)
                    # Select global det. points for current camera
                    temp_glob_det = self.global_detections[:,self.global_detections[0] == i]
                    # Save traj. point locations in global_traj
                    #temp_glob_det_mask = np.isin(self.global_detections[2],temp_glob_det[2])
                    temp_glob_det_mask = np.where(self.global_detections[0] == i)
                    # Save camera detections that are used within the global traj. 
                    _,temp_glob_traj_idx,temp_glob_traj_det_idx = np.intersect1d(temp_glob_traj[2],self.detections[i][0],return_indices=True,assume_unique=True)
                    # Save camera detections that are used within global detections. 
                    _,temp_glob_det_idx,temp_det_idx = np.intersect1d(temp_glob_det[1],self.detections[i][0],return_indices=True,assume_unique=True)  
                    
                    #assert np.sum(temp_glob_traj_mask == True) == len(temp_glob_traj_det_idx)
                    assert np.shape(temp_glob_traj_mask)[1] == np.shape(temp_glob_traj_det_idx)[0]
                    assert np.shape(temp_glob_det_mask)[1] == np.shape(self.detections_global[i][0])[0]
                    
                    # Update global detection timestamps for cam_id
                    self.global_detections[2,temp_glob_det_mask] = self.detections_global[i][0]
                    # Update global traj timestamps for detections in global_traj
                    self.global_traj[3,temp_glob_traj_mask] = self.detections_global[i][0,temp_glob_traj_det_idx]
        if motion_prior:
            # Resort global_traj according to updated global timestamps 
            if not (self.global_traj[3,1:]>=self.global_traj[3,:-1]).all():
                self.global_traj[:,np.argsort(self.global_traj[3,:])] 


    def cut_detection(self,second=1):
        '''
        Truncate the starting and end part of each continuous part of the detections
        '''

        if not second: return

        for i in range(self.numCam):
            detect = self.detections[i]
            interval = util.find_intervals(detect[0])
            cut = int(self.cameras[i].fps * second)

            interval_long = interval[:,interval[1]-interval[0]>cut*2]
            interval_long[0] += cut
            interval_long[1] -= cut

            assert (interval_long[1]-interval_long[0]>=0).all()

            self.detections[i], _ = util.sampling(detect,interval_long)


    def init_traj(self,error=10,inlier_only=False):
        '''
        Select the first two cams in the sequence, compute fundamental matrix, triangulate points
        '''

        self.select_most_overlap(init=True)

        t1, t2 = self.sequence[0], self.sequence[1]
        K1, K2 = self.cameras[t1].K, self.cameras[t2].K

        # Find correspondences
        if self.cameras[t1].fps > self.cameras[t2].fps:
            d1, d2 = util.match_overlap(self.detections_global[t1], self.detections_global[t2])
        else:
            d2, d1 = util.match_overlap(self.detections_global[t2], self.detections_global[t1])
        
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


    def traj_to_spline(self,smooth_factor):
        '''
        Convert discrete 3D trajectory into spline representation

        A single spline is built for each interval
        '''

        assert len(smooth_factor)==2, 'Smoothness should be defined by two parameters (min, max)'

        timestamp = self.traj[0]
        interval, idx = util.find_intervals(timestamp,idx=True)
        tck = [None] * interval.shape[1]

        for i in range(interval.shape[1]):
            part = self.traj[:,idx[0,i]:idx[1,i]+1]

            measure = part[0,-1] - part[0,0]
            s = (1e-3)**2*measure
            thres_min, thres_max = min(smooth_factor), max(smooth_factor)
            prev = 0
            t = 0
            try:
                while True:
                    tck[i], u = interpolate.splprep(part[1:],u=part[0],s=s,k=3)

                    numKnot = len(tck[i][0])-4
                    if numKnot == prev and numKnot==4 and t==2:
                        break
                    else:
                        prev = numKnot
                    
                    if measure/numKnot > thres_max:
                        s /= 1.5
                        t = 1
                    elif measure/numKnot < thres_min:
                        s *= 2
                        t = 2
                    else:
                        break

                dist = np.sum(np.sqrt(np.sum((part[1:,1:]-part[1:,:-1])**2,axis=0)))

            except:
                tck[i], u = interpolate.splprep(part[1:],u=part[0],s=s,k=1)
            
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

        assert (self.traj[0,1:] >= self.traj[0,:-1]).all()

        return self.traj


    def error_cam(self,cam_id,mode='dist',motion_prior=False,norm=False):
        '''
        Calculate the reprojection errors for a given camera

        Different modes are available: 'dist', 'xy_1D', 'xy_2D', 'each'
        '''

        tck, interval = self.spline['tck'], self.spline['int']
        if motion_prior:
            self.detection_to_global(motion_prior=motion_prior)
        else:
            self.detection_to_global(cam_id)

        _, idx = util.sampling(self.detections_global[cam_id], interval, belong=True)
        detect = np.empty([3,0])
        point_3D = np.empty([3,0])
        for i in range(interval.shape[1]):
            detect_part = self.detections_global[cam_id][:,idx==i+1]
            if detect_part.size:
                if motion_prior:
                    cam_global_traj = self.global_traj[:,self.global_traj[1] == cam_id]
                    _,traj_idx,detect_idx = np.intersect1d(cam_global_traj[3],detect_part[0],assume_unique=True,return_indices=True)
                    detect_part = detect_part[:,detect_idx]
                    detect = np.hstack((detect,detect_part))
                    point_3D = np.hstack((point_3D,cam_global_traj[4:,traj_idx]))
                else:
                    detect = np.hstack((detect,detect_part)) 
                    point_3D = np.hstack((point_3D, np.asarray(interpolate.splev(detect_part[0], tck[i]))))
                
        X = util.homogeneous(point_3D)
        x = detect[1:]
        x_cal = self.cameras[cam_id].projectPoint(X)
        
        #Normalize Tracks
        if norm:
            x_cal = np.dot(np.linalg.inv(self.cameras[cam_id].K), x_cal)
            x = np.dot(np.linalg.inv(self.cameras[cam_id].K), util.homogeneous(x))

        if mode == 'dist':
            return ep.reprojection_error(x, x_cal)
        elif mode == 'xy_1D':
            return np.concatenate((abs(x_cal[0]-x[0]),abs(x_cal[1]-x[1])))
        elif mode == 'xy_2D':
            return np.vstack((abs(x_cal[0]-x[0]),abs(x_cal[1]-x[1])))
        elif mode == 'each':
            error_x = np.zeros_like(self.detections[cam_id][0])
            error_y = np.zeros_like(self.detections[cam_id][0])
            if motion_prior:
                _,det_idx,_ = np.intersect1d(self.detections_global[cam_id][0],detect[0],assume_unique=True,return_indices=True)
                assert det_idx.shape[0] == x_cal.shape[1], '# of detections and traj. points are not equal'
                error_x[det_idx] = abs(x_cal[0]-x[0])
                error_y[det_idx] = abs(x_cal[1]-x[1])
            else:
                error_x[idx.astype(bool)] = abs(x_cal[0]-x[0])
                error_y[idx.astype(bool)] = abs(x_cal[1]-x[1])
            return np.concatenate((error_x, error_y))
    

    def error_motion(self,cams,mode='dist',norm=False,motion_weights=0,motion_reg = False,motion_prior = False):
        '''
        Calculate the reprojection errors for a given camera for a multi_spline object. 

        - Accounts for motion prior

        Different modes are available: 'dist', 'xy_1D', 'xy_2D', 'each'

        computes error for motion prior regularization
        '''

        interval = self.spline['int']
        
        # Update global_detections and global_traj timestamps
        self.detection_to_global(cams,motion_prior=True)
        # Update global_traj for motion_reg
        if motion_reg:
            self.spline_to_traj()
            _, idx = util.sampling(self.traj[0], interval, belong=True)
        if motion_prior:
            _, idx = util.sampling(self.global_traj[3], interval, belong=True)

        detect = np.empty([3,0])
        point_3D = np.empty([3,0])
        temp_glob_ts = np.array([])
        mot_err_res = np.array([])
         

        if motion_prior:
            global_traj_ts = np.array([])
            for i in range(interval.shape[1]):
                traj_part = self.global_traj[:,idx==i+1]
                if traj_part.size:
                    weights = np.ones(traj_part.shape[1]) * motion_weights
                    mot_err = self.motion_prior(traj_part[3:],weights,prior=self.settings['motion_type'])
                    mot_err_res = np.concatenate((mot_err_res, mot_err))
                    if self.settings['motion_type'] == 'F':
                        global_traj_ts = np.concatenate((global_traj_ts, traj_part[3,1:-1]))
                    else:
                        global_traj_ts = np.concatenate((global_traj_ts, traj_part[3,1:]))
            motion_error = np.zeros((self.global_traj.shape[1]))
            _,traj_idx,_ = np.intersect1d(self.global_traj[3],global_traj_ts,assume_unique=True,return_indices=True)
            assert traj_idx.shape[0] == mot_err_res.shape[0], 'wrong number of global_traj points'
            motion_error[traj_idx] = mot_err_res
        
        elif motion_reg :
            traj_ts = np.array([]) 
            motion_error = np.zeros((self.traj.shape[1]))
            for i in range(interval.shape[1]):
                traj_part = self.traj[:,idx==i+1]
                if traj_part.size:
                    weights = np.ones(traj_part.shape[1]) * motion_weights
                    mot_err = self.motion_prior(traj_part,weights,prior=self.settings['motion_type'])
                    mot_err_res = np.concatenate((mot_err_res, mot_err))
                    if self.settings['motion_type'] == 'F':
                        traj_ts = np.concatenate((traj_ts, traj_part[0,1:-1]))  
                    else:
                        traj_ts = np.concatenate((traj_ts, traj_part[0,1:])) 
            _,traj_idx,_ = np.intersect1d(self.traj[0],traj_ts,assume_unique=True,return_indices=True)
            motion_error[traj_idx] = mot_err_res
            
        return motion_error
    

    def compute_visibility(self):
        '''
        Decide for each raw detection if it is visible from current 3D spline
        '''

        self.visible = []
        interval = self.spline['int']
        self.detection_to_global()

        for cam_id in range(self.numCam):
            _, visible = util.sampling(self.detections_global[cam_id], interval, belong=True)
            self.visible.append(visible)


    def BA(self, numCam, max_iter=10, rs=False, motion_prior=False,motion_reg=False,motion_weights=1,norm=False):
        '''
        Bundle Adjustment with multiple splines

        The camera order is assumed to be the same as self.sequence
        '''

        def error_BA(x):
            '''
            Input is the model (parameters that need to be optimized)
            '''

            # Assign parameters to the class attributes
            sections = [numCam, numCam*2, numCam*3, numCam*3+numCam*num_camParam]
            model_parts = np.split(x, sections)
            self.alpha[self.sequence[:numCam]], self.beta[self.sequence[:numCam]], self.rs[self.sequence[:numCam]] = model_parts[0], model_parts[1], model_parts[2]

            cams = np.split(model_parts[3],numCam)
            for i in range(numCam):
                self.cameras[self.sequence[i]].vector2P(cams[i], calib=self.settings['opt_calib']) 
            
            if motion_reg:
                #interpolate 3d points from detections in all cameras
                self.all_detect_to_traj(self.sequence[:numCam])
                
            if motion_prior:
                self.global_traj[4:] = model_parts[4].reshape(-1,3).T
            
            else:
                spline_parts = np.split(model_parts[4],idx_spline[0,1:])
                for i in range(len(spline_parts)):
                    spline_i = spline_parts[i].reshape(3,-1)
                    self.spline['tck'][i][1] = [spline_i[0],spline_i[1],spline_i[2]]
            
            # Compute errors
            error = np.array([])
            for i in range(numCam):
                error_each = self.error_cam(self.sequence[i], mode='each')
                error = np.concatenate((error, error_each))
            if motion_prior:
                error_motion_prior = self.error_motion(self.sequence[:numCam],motion_weights=motion_weights,motion_prior=True)
                error = np.concatenate((error, error_motion_prior))
            if motion_reg:
                error_motion_reg = self.error_motion(self.sequence[:numCam],motion_reg=True,motion_weights=motion_weights)
                error = np.concatenate((error, error_motion_reg))
            
            return error


        def jac_BA(near=3,motion_offset=10):

            num_param = len(model)
            self.compute_visibility()

            jac = lil_matrix((1, num_param),dtype=int)
            #jac = np.empty([0,num_param])

            if motion_reg:
                    m_jac = lil_matrix((self.traj.shape[1], num_param),dtype=int)
            elif motion_prior:
                m_jac = lil_matrix((self.global_traj.shape[1], num_param),dtype=int)
            
            for i in range(numCam):
                cam_id = self.sequence[i]
                num_detect = self.detections[cam_id].shape[1]

                # consider only reprojection in x direction, which is the same in y direction
                jac_cam = lil_matrix((num_detect, num_param),dtype=int)
                #jac_cam = np.zeros((num_detect, num_param))

                # alpha and beta
                jac_cam[:,[i,i+numCam]] = 1

                # rolling shutter
                if rs:
                    jac_cam[:,i+numCam*2] = 1
                else:
                    jac_cam[:,i+numCam*2] = 0

                # camera parameters
                start = 3*numCam+i*num_camParam
                jac_cam[:,start:start+num_camParam] = 1

                if motion_prior:
                    traj_start = numCam * (3+num_camParam)
                    traj_len = self.global_traj.shape[1]
                    for j in range(num_detect):
                        # Verify traj. point lies within current spline interval
                        if self.visible[cam_id][j]:
                            timestamp = self.detections_global[cam_id][0,j]
                            traj_pnt = np.where(self.global_traj[3] == timestamp)[0]
                            traj_pnt += traj_start
                            if (traj_pnt-traj_start) < motion_offset:
                                traj_idx = np.arange(traj_start,traj_pnt+motion_offset)   
                            else:
                                traj_idx = np.arange(traj_pnt-motion_offset,traj_pnt+motion_offset) 
                                
                            traj_idx = np.concatenate((traj_idx, traj_idx+traj_len, traj_idx+2*traj_len))
                            
                            if np.array(traj_idx < num_param).all():
                                jac_cam[j,traj_idx] = 1 
                            else:
                                jac_cam[j,traj_idx[traj_idx < num_param]] = 1         
                        else:
                            jac_cam[j] = 0
                        
                    jac = vstack((jac, vstack([jac_cam,jac_cam])))
                # spline parameters
                else:
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

                    jac = vstack((jac, vstack([jac_cam,jac_cam])))
                    #jac = np.vstack((jac, np.tile(jac_cam,(2,1))))

            if motion_reg:
                tck, interval = self.spline['tck'], self.spline['int']
                for j in range(self.traj.shape[1]):
                    _, spline_id = util.sampling(self.traj[:,j], interval, belong=True)
                    detect = np.empty([3,0])
                    point_3D = np.empty([3,0])
                    
                    # Find the corresponding spline for each interpolated point
                    spline_id[0] -= 1
                    knot = self.spline['tck'][spline_id[0]][0][2:-2]
                    timestamp = self.traj[0,j]
                    knot_idx = np.argsort(abs(knot-timestamp))[:near]
                    knot_idx = np.concatenate((knot_idx, knot_idx+len(knot), knot_idx+2*len(knot)))
                    m_jac[j,idx_spline_sum[0,spline_id[0]]+knot_idx] = 1
                jac = vstack((jac, m_jac))
            
            elif motion_prior:
                m_jac = lil_matrix((self.global_traj.shape[1], num_param),dtype=int)
                traj_start = numCam * (3+num_camParam)
                for j in range(self.global_traj.shape[1]):
                        m_jac[j] = 0
                        if j < motion_offset:
                           m_traj_idx = np.arange(0,j+motion_offset) 
                           m_traj_idx += traj_start#
                        else:
                            m_traj_idx = np.arange(j-motion_offset,j+motion_offset) 
                            m_traj_idx += traj_start
                        m_traj_idx = np.concatenate((m_traj_idx, m_traj_idx+traj_len, m_traj_idx+2*traj_len))
                        
                        if np.array(m_traj_idx < num_param).all():
                            m_jac[j,m_traj_idx] = 1
                        else:
                            m_jac[j,m_traj_idx[m_traj_idx < num_param]] = 1
                
                jac = vstack((jac, m_jac))
                
            # fix the first camera
            # jac[:,[0,numCam]], jac[:,2*numCam+4:2*numCam+10] = 0, 0
            #return jac
            return jac.toarray()[1:]

        starttime = datetime.now()
        
        '''Before BA'''
        # Define Parameters that will be optimized
        model_alpha = self.alpha[self.sequence[:numCam]]
        model_beta = self.beta[self.sequence[:numCam]]
        model_rs = self.rs[self.sequence[:numCam]]

        model_cam = np.array([])
        num_camParam = 15 if self.settings['opt_calib'] else 6
        for i in self.sequence[:numCam]:
            model_cam = np.concatenate((model_cam, self.cameras[i].P2vector(calib=self.settings['opt_calib'])))

        model_other = np.concatenate((model_alpha, model_beta, model_rs, model_cam))
        if motion_prior:
            #interpolate 3d points from detections in all cameras
            self.all_detect_to_traj(self.sequence[:numCam])
        if motion_reg:
            self.spline_to_traj()
        if motion_prior:
            #interpolate 3d points from detections in all cameras
            model_traj = np.ravel(self.global_traj[4:].T)
            model = np.concatenate((model_other, model_traj))
        else:
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

            idx_spline_sum = idx_spline + len(model_other)
            model = np.concatenate((model_other, model_spline))
            assert idx_spline_sum[-1,-1] == len(model), 'Error in spline indices'
        print('Number of BA parameters is {}'.format(len(model)))

        # Set the Jacobian matrix
        A = jac_BA()

        '''Compute BA'''
        print('Doing BA with {} cameras...\n'.format(numCam))
        fn = lambda x: error_BA(x)
        res = least_squares(fn,model,jac_sparsity=A,tr_solver='lsmr',xtol=1e-12,max_nfev=max_iter,verbose=0)

        '''After BA'''
        # Assign the optimized model to alpha, beta, cam, and spline
        sections = [numCam, numCam*2, numCam*3, numCam*3+numCam*num_camParam]
        model_parts = np.split(res.x, sections)
        self.alpha[self.sequence[:numCam]], self.beta[self.sequence[:numCam]], self.rs[self.sequence[:numCam]] = model_parts[0], model_parts[1], model_parts[2]
        
        cams = np.split(model_parts[3],numCam)
        for i in range(numCam):
            self.cameras[self.sequence[i]].vector2P(cams[i], calib=self.settings['opt_calib']) 
        if motion_prior:
            self.global_traj[4:] = model_parts[4].reshape(-1,3).T
            if (self.global_traj[3][1:]>self.global_traj[3][:-1]).all():
                self.traj_to_spline(smooth_factor=self.settings['smooth_factor'])
            else:
                self.traj = self.global_traj[3:,np.argsort(self.global_traj[3,:])] 
                self.traj_to_spline(smooth_factor=self.settings['smooth_factor'])
        else:
            spline_parts = np.split(model_parts[4],idx_spline[0,1:])
            for i in range(len(spline_parts)):
                spline_i = spline_parts[i].reshape(3,-1)
                self.spline['tck'][i][1] = [spline_i[0],spline_i[1],spline_i[2]]

        # Update global timestamps for each serie of detections
        self.detection_to_global()

        return res


    def remove_outliers(self, cams, thres=30, verbose=False):
        '''
        Remove raw detections that have large reprojection errors.

        But the 3D spline won't be changed
        '''

        if thres:
            for i in cams:
                error_all = self.error_cam(i,mode='each')
                error_xy = np.split(error_all,2)
                error = np.sqrt(error_xy[0]**2 + error_xy[1]**2)

                self.detections[i] = self.detections[i][:,error<thres]
                self.detection_to_global(i)

                if verbose:
                    print('{} out of {} detections are removed for camera {}'.format(sum(error>=thres),sum(error!=0),i))


    def get_camera_pose(self, cam_id, error=8, verbose=0):
        '''
        Get the absolute pose of a camera by solving the PnP problem.

        Take care with DISTORSION model!
        '''
        
        tck, interval = self.spline['tck'], self.spline['int']
        self.detection_to_global(cam_id)

        _, idx = util.sampling(self.detections_global[cam_id], interval, belong=True)
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
        distCoeffs = self.cameras[cam_id].d
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, self.cameras[cam_id].K, distCoeffs, reprojectionError=error)

        self.cameras[cam_id].R = cv2.Rodrigues(rvec)[0]
        self.cameras[cam_id].t = tvec.reshape(-1,)
        self.cameras[cam_id].compose()

        if verbose:
            print('{} out of {} points are inliers for PnP'.format(inliers.shape[0], N))
            

    def triangulate(self, cam_id, cams, factor_t2s, factor_s2t=0.02, thres=0, refit=True, verbose=0):
        '''
        Triangulate new points to the existing 3D spline and optionally refit it

        cam_id is the new camera
        
        cams must be an iterable that contains cameras that have been processed to build the 3D spline
        '''

        assert self.cameras[cam_id].P is not None, 'The camera pose must be computed first'
        tck, interval = self.spline['tck'], self.spline['int']
        self.detection_to_global(cam_id)

        # Find detections from this camera that haven't been triangulated yet
        _, idx_ex = util.sampling(self.detections_global[cam_id], interval)
        detect_new = self.detections_global[cam_id][:, np.logical_not(idx_ex)]

        # Matching these detections with detections from previous cameras and triangulate them
        X_new = np.empty([4,0])
        for i in cams:
            self.detection_to_global(i)
            detect_ex = self.detections_global[i]

            # Detections of previous cameras are interpolated, no matter the fps
            try:
                x1, x2 = util.match_overlap(detect_new, detect_ex)
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

        _, idx_empty = util.sampling(X_new, interval)
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
            detect_i, _ = util.sampling(self.detections_global[i],interval)
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
                plt.suptitle('Camera {}: undistorted detection (red) vs reprojection (blue)'.format(i))

        plt.show()


    def select_most_overlap(self,init=False):
        '''
        Select either the initial pair of cameras or the next best camera with largest overlap
        '''

        if not self.find_order:
            return

        self.detection_to_global()
        overlap_max = 0
        
        if init:
            for i in range(self.numCam-1):
                for j in range(i+1,self.numCam):
                    x, y = util.match_overlap(self.detections_global[i],self.detections_global[j])
                    overlap = x.shape[1] / self.cameras[i].fps
                    if overlap > overlap_max:
                        overlap_max = overlap
                        init_pair = [i,j]
            self.sequence = init_pair
        else:
            traj = self.spline_to_traj()
            candidate = []
            for i in range(self.numCam):
                if self.cameras[i].P is None:
                    candidate.append(i)
            for i in candidate:
                interval = util.find_intervals(self.detections_global[i][0])
                overlap, _ = util.sampling(traj[0], interval)

                if len(overlap) > overlap_max:
                    overlap_max = len(overlap)
                    next_cam = i
            self.sequence.append(next_cam)


    def all_detect_to_traj(self,*cam):
        #global_traj = np.empty()
        global_time_stamps_all = np.array([])
        frame_id_all = np.array([])
        cam_id = np.array([])

        if len(cam):
            for i in cam[0]:
                self.detection_to_global(i)
                global_time_stamps_all = np.concatenate((global_time_stamps_all,self.detections_global[i][0]))
                frame_id_all  = np.concatenate((frame_id_all,self.detections[i][0]))
                cam_id = np.concatenate((cam_id,np.ones(len(self.detections[i][0])) * i ))

        else:
            for i in range(self.numCam):
                self.detection_to_global(i)
                global_time_stamps_all = np.concatenate((global_time_stamps_all,self.detections_global[i][0]))
                frame_id_all = np.concatenate((frame_id_all,self.detections[i][0]))
                cam_id = np.concatenate((cam_id,np.ones(len(self.detections[i][0])) * i ))

        self.frame_id_all = frame_id_all 
        #global_time_stamps_all = np.sort(global_time_stamps_all)
        #Remove duplicate timestamps
        #if (global_time_stamps_all[1:]==global_time_stamps_all[:-1]).any():
        #    gt_uniq = remove_dupes(global_time_stamps_all)
        #    fram_id_uniq = remove_dupes(frame_id_all)
        #    gt_uniq = global_time_stamps_all[1:][(global_time_stamps_all[1:]>global_time_stamps_all[:-1])]
        #    gt_dupes = global_time_stamps_all[1:][(global_time_stamps_all[1:] == global_time_stamps_all[:-1])]
        #    gt_uniq = np.hstack((global_time_stamps_all[0],gt_uniq))
        #    assert (gt_uniq[1:]>gt_uniq[:-1]).all()
        #    self.global_time_stamps_all = gt_uniq #global_time_stamps_all
        #else:
        self.global_time_stamps_all = global_time_stamps_all

        #assert (global_time_stamps_all[1:]>=global_time_stamps_all[:-1]).all()
        
        
        # Interpolate 3D points for global timestamps in all cameras
        self.spline_to_traj(t=np.sort(global_time_stamps_all))

        self.global_detections = np.vstack((cam_id,frame_id_all,global_time_stamps_all))
        #Sort global_traj by global time stamp
        temp_global_traj = self.global_detections[:,np.argsort(self.global_detections[2,:])]
        
        # Remove duplicate timestamps from temp traj.
        #temp_global_traj = np.hstack((temp_global_traj[:,0].reshape(3,1),temp_global_traj[:,1:][:,(temp_global_traj[2][1:]>temp_global_traj[2][:-1])]))
        
        # Remove duplicate timestamps from self.traj.
        #if (self.traj[0][1:]==self.traj[0][:-1]).any():
        #    self.traj = np.hstack((self.traj[:,0].reshape(4,1),self.traj[:,1:][:,(self.traj[0][1:]>self.traj[0][:-1])]))
        
        # Create ascending global timestamp trajectory
        #_,traj_idx,_= np.intersect1d(temp_global_traj[2],self.traj[0],assume_unique=False,return_indices=True)
        traj_idx = np.isin(temp_global_traj[2],self.traj[0])
        temp_global_traj = np.vstack((temp_global_traj[:,traj_idx],self.traj[1:]))
        # Apply index to track original order of the global traj.
        temp_global_traj = np.vstack((np.arange(temp_global_traj.shape[1]),temp_global_traj))
        self.global_traj = temp_global_traj
        
        #verify global timestamps are sorted in ascending order
        assert (self.global_traj[3][1:]>=self.global_traj[3][:-1]).all(), 'timestamps are not in ascending order'
        
        # # Plot timestamps for visualinspection
        # fig = plt.figure(figsize=(12, 10))
        # num = self.global_traj.shape[1]
        # ax = fig.add_subplot(1,1,1)
        # ax.scatter(np.arange(num),x[3])
        # #ax.scatter(np.arange(num),x[2]) #,c=np.arange(x[i].shape[1])*color)
        # plt.show()
        #vis.show_trajectory_3D(self.global_traj[4:],color=None)

    
    def motion_prior(self,traj,weights,eps=1e-20,prior='F'):
        
        '''
        Function defining the physical motion constraint for the triangulated trajectory.

        inputs: 
        
        weighting: factor defined by the 2D reprojection uncertainty
        X: 3D point sequence

        returns: 
        Sm - cost function for physical motion prior
        
        '''
        assert traj.shape[0]==4, '3D points must be of shape 4 x n where row 0 is the time index'
        # Constant Kinetic Energy Motion Prior
        ts = traj[0]
        if prior == 'KE':
            
            traj_for = traj[1:,:-1]
            traj_aft = traj[1:,1:]
            vel = (traj_aft - traj_for)/((ts[1:]-ts[:-1])+eps)
            mot_resid = np.array([weights[:traj_for.shape[1]]*0.5*(vel**2 * (ts[1:]-ts[:-1]))])
        
        # Constant Force Motion Prior
        if prior == 'F':
            
            traj_for = traj[1:,:-2]
            traj_mid = traj[1:,1:-1]
            traj_aft = traj[1:,2:]

            dt1 = ts[1:-1] - ts[:-2]
            dt2 = ts[2:] - ts[1:-1]
            dt3 = dt1 + dt2 

            v1 = (traj_mid - traj_for) / ( dt1 + eps)
            v2 = (traj_aft - traj_mid) / ( dt2 + eps )

            accel = (v2 - v1) / (dt3 + eps)
            mot_resid = np.array([weights[:traj_for.shape[1]]*(accel * (dt3))])

        mot_resid = np.sum(abs(mot_resid[0]),axis=0)
        return mot_resid
        
        
    def time_shift(self, iter=False):
        '''
        This function computes relative time shifts of each camera to the ref camera using the given corresponding frame numbers

        If the given frame indices are precise, then the time shifts are directly transformed from them.
        '''

        assert len(self.cf)==self.numCam, 'The number of frame indices should equal to the number of cameras'

        if self.settings['cf_exact']:
            self.beta = self.cf[self.ref_cam] - self.alpha*self.cf
            print('The given corresponding frames are directly exploited as temporal synchronization\n')
        else:

            if self.settings['sync_method'] == 'iter':
                sync_fun = sync.sync_iter
            elif self.settings['sync_method'] == 'bf':
                sync_fun = sync.sync_bf
            else:
                raise ValueError('Synchronization method must be either "iter" or "bf"')

            print('Computing temporal synchronization...\n')
            beta = np.zeros(self.numCam)
            i = self.ref_cam
            for j in range(self.numCam):
                if j==i:
                    beta[j] = 0
                else:
                    beta[j], _ = sync_fun(self.cameras[i].fps, self.cameras[j].fps,
                                                    self.detections[i], self.detections[j],
                                                    self.cf[i], self.cf[j])
                print('Status: {} from {} cam finished'.format(j+1,self.numCam))
            self.beta = beta


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
        self.resolution = kwargs.get('resolution')


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


    def P2vector(self, calib=False):
        '''
        Convert camera parameters into a vector
        '''

        k = np.array([self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]])
        r = cv2.Rodrigues(self.R)[0].reshape(-1,)

        if calib:
            return np.concatenate((k, r, self.t, self.d))
        else:
            return np.concatenate((r,self.t))


    def vector2P(self, vector, calib=False):
        '''
        Convert a vector into camera parameters
        '''

        if calib:
            self.K = np.diag((1,1,1)).astype(float)
            self.K[0,0], self.K[1,1] = vector[0], vector[1]
            self.K[:2,-1] = vector[2:4]
            self.R = cv2.Rodrigues(vector[4:7])[0]
            self.t = vector[7:10]
            self.d = vector[10:]
        else:
            self.R = cv2.Rodrigues(vector[:3])[0]
            self.t = vector[3:6]

        self.compose()
        return self.P
    

    def undist_point(self,points):
        
        assert points.shape[0]==2, 'Input must be a 2D array'

        num = points.shape[1]

        src = np.ascontiguousarray(points.T).reshape((num,1,2))
        dst = cv2.undistortPoints(src, self.K, self.d)
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


def create_scene(path_input):
    '''
    Create a scene from the imput template in json format
    '''

    # Read the config file
    with open(path_input, 'r') as file:
        config = json.load(file)

    # Create the scene
    flight = Scene()

    # Load settings
    flight.settings = config['settings']

    # Load detections
    path_detect = config['necessary inputs']['path_detections']
    flight.numCam = len(path_detect)
    for i in path_detect:
        detect = np.loadtxt(i,usecols=(2,0,1))[:flight.settings['num_detections']].T
        flight.addDetection(detect)

    # Load cameras
    path_cam = config['necessary inputs']['path_cameras']
    for path in path_cam:
        try:
            with open(path, 'r') as file:
                cam = json.load(file)
        except:
            raise Exception('Wrong input of camera')

        if len(cam['distCoeff']) == 4:
            cam['distCoeff'].append(0)
        
        flight.addCamera(Camera(K=np.asfarray(cam['K-matrix']), d=np.asfarray(cam['distCoeff']),
                                fps=cam['fps'], resolution=cam['resolution']))

    #  Load sequence
    flight.ref_cam = config['settings']['ref_cam']
    flight.sequence = config['settings']['camera_sequence']
    flight.find_order = False if len(flight.sequence) else True

    # Load corresponding frames
    flight.cf = np.asfarray(config['necessary inputs']['corresponding_frames'])

    # Load rolling shutter parameter
    init_rs = config['settings']['init_rs'] if config['settings']['rolling_shutter'] else 0
    if isinstance(init_rs,list):
        assert len(init_rs) == flight.numCam, 'the number of initial rolling shutter values must equal the number of cameras'
        flight.rs = np.asfarray([init_rs[i] for i in range(flight.numCam)])
    else:
        flight.rs = np.asfarray([init_rs for i in range(flight.numCam)])

    # Load ground truth setting (optinal)
    flight.gt = config['optional inputs']['ground_truth']

    print('Input data are loaded successfully, a scene is created.\n')
    return flight
