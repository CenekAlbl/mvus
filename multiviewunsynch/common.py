# Classes that are common to the entire project
import numpy as np
import util
import epipolar as ep
import synchronization
import scipy.io as scio
import pickle
import copy
import cv2
import visualization as vis
from datetime import datetime
from scipy.optimize import least_squares


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
            self.sequence = self.beta[0].argsort()[::-1]


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

        v = np.zeros((self.numCam,self.traj.shape[1]))
        for i in range(self.numCam):
            inter,idx1,idx2 = np.intersect1d(self.traj[0],self.tracks[i][0],assume_unique=True,return_indices=True)
            v[i,idx1] = 1
        self.visible = v

        # self.numPoint = 0
        # for i in self.tracks:
        #     if i[0,-1] > self.numPoint:
        #         self.numPoint = int(i[0,-1])
        # self.numPoint+=1

        # self.visible = np.zeros((self.numCam,self.numPoint))
        # self.jac = np.zeros((self.numPoint*self.numCam*2,self.numCam*11+self.numPoint*3))
        # for i in range(self.numCam):
        #     _,idx,_ = np.intersect1d(np.arange(self.numPoint),self.tracks[i][0],assume_unique=True,return_indices=True)
        #     self.visible[i,idx]=1
        #     for j in self.tracks[i][0].astype(int):
        #         self.jac[(self.numPoint*2*i+j, self.numPoint*(2*i+1)+j),i*11:(i+1)*11] = 1
        #         self.jac[(self.numPoint*2*i+j, self.numPoint*(2*i+1)+j),11*self.numCam+j*3:11*self.numCam+(j+1)*3] = 1
                

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

    
    def error_cam(self,id_cam):
        '''
        This function computes the reprojection error for a single camera, such that only visible 3D points will be reprojected
        '''

        inter,idx1,idx2 = np.intersect1d(self.traj[0],self.tracks[id_cam][0],assume_unique=True,return_indices=True)
        X = util.homogeneous(self.traj[1:,idx1])
        x = util.homogeneous(self.tracks[id_cam][1:,idx2])
        
        x_cal = self.cameras[id_cam].projectPoint(X)
        error = ep.reprojection_error(x,x_cal)
        print("Mean reprojection error for camera {} is: ".format(id_cam), np.mean(error))
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


    def discard_fn(self):
        '''
        Functions that are not needed (for now) are stored here
        '''
        def optimize_traj(self):
            '''
            This function is no longer used, because polynomial triangulation is applied
            '''

            def error_fn(model):

                self.traj[1:] = model.reshape(3,-1)

                cam_1, cam_2  = self.cameras[self.sequence[0]], self.cameras[self.sequence[1]]
                Track = [self.tracks[self.sequence[0]],self.tracks[self.sequence[1]]]
                X = self.traj

                # error from Camera 1
                inter,idx1,idx2 = np.intersect1d(X[0],Track[0][0],assume_unique=True,return_indices=True)
                X_temp = util.homogeneous(X[1:,idx1])
                x = Track[0][1:,idx2]
                x_repro = cam_1.projectPoint(X_temp)
                err_x1,err_y1 = abs(x[0]-x_repro[0]), abs(x[1]-x_repro[1])

                # error from Camera 2
                inter,idx1,idx2 = np.intersect1d(X[0],Track[1][0],assume_unique=True,return_indices=True)
                X_temp = util.homogeneous(X[1:,idx1])
                x = Track[1][1:,idx2]
                x_repro = cam_2.projectPoint(X_temp)
                err_x2,err_y2 = abs(x[0]-x_repro[0]), abs(x[1]-x_repro[1])

                error = np.concatenate((err_x1,err_y1,err_x2,err_y2))

                # Make sure the error vector has a fixed size
                if len(err_x1) < len(self.traj[0]):
                    error = np.concatenate((error,np.zeros(2*(len(self.traj[0])-len(err_x1)))))
                if len(err_x2) < len(self.traj[0]):
                    error = np.concatenate((error,np.zeros(2*(len(self.traj[0])-len(err_x2)))))

                return error
            
            # before
            model = np.ravel(self.traj[1:])

            # during
            fn = lambda x: error_fn(x)
            res = least_squares(fn,model)

            # after
            self.traj[1:] = res.x.reshape(3,-1)
            
            return res


        def optimize_two_Rt(self,c2):
            '''
            This function assumes that the first camera is the one started first, c2 is the index for the second camera

            Only R and t will be optimized, K and distortion fixed
            '''

            def error_fn(model):

                self.beta[0,c2] = model[0]
                self.cameras[c1].vector2Rt(model[1:7])
                self.cameras[c2].vector2Rt(model[7:13])
                self.set_tracks()

                cam_1, cam_2  = self.cameras[c1], self.cameras[c2]
                Track = [self.tracks[c1],self.tracks[c2]]
                X = np.vstack((self.traj[0], model[13:].reshape(3,-1)))

                # error from Camera 1
                inter,idx1,idx2 = np.intersect1d(X[0],Track[0][0],assume_unique=True,return_indices=True)
                X_temp = util.homogeneous(X[1:,idx1])
                x = Track[0][1:,idx2]
                x_repro = cam_1.projectPoint(X_temp)
                err_x1,err_y1 = abs(x[0]-x_repro[0]), abs(x[1]-x_repro[1])

                # error from Camera 2
                inter,idx1,idx2 = np.intersect1d(X[0],Track[1][0],assume_unique=True,return_indices=True)
                X_temp = util.homogeneous(X[1:,idx1])
                x = Track[1][1:,idx2]
                x_repro = cam_2.projectPoint(X_temp)
                err_x2,err_y2 = abs(x[0]-x_repro[0]), abs(x[1]-x_repro[1])

                error = np.concatenate((err_x1,err_y1,err_x2,err_y2))

                # Make sure the error vector has a fixed size
                if len(err_x1) < len(self.traj[0]):
                    error = np.concatenate((error,np.zeros(2*(len(self.traj[0])-len(err_x1)))))
                if len(err_x2) < len(self.traj[0]):
                    error = np.concatenate((error,np.zeros(2*(len(self.traj[0])-len(err_x2)))))

                return error
            
            # before
            c1 = 0
            model = np.array([self.beta[0,c2]])
            model = np.concatenate((model,self.cameras[c1].Rt2vector(),self.cameras[c2].Rt2vector()))
            model = np.concatenate((model,np.ravel(self.traj[1:])))

            # during
            fn = lambda x: error_fn(x)
            res = least_squares(fn,model)

            # after
            self.beta[0,c2] = res.x[0]
            self.cameras[c1].vector2Rt(res.x[1:7])
            self.cameras[c2].vector2Rt(res.x[7:13])
            self.set_tracks()
            self.traj[1:] = res.x[13:].reshape(3,-1)
            
            return res


        def optimize_two_KRt(self,c2):
            '''
            This function assumes that the first camera is the one started first, c2 is the index for the second camera

            K, R, t are optimized
            '''

            def error_fn(model):

                self.beta[0,c2] = model[0]
                self.cameras[c1].vector2P_9(model[1:10])
                self.cameras[c2].vector2P_9(model[10:19])
                self.set_tracks()

                Track = [self.tracks[c1],self.tracks[c2]]
                X = np.vstack((self.traj[0], model[19:].reshape(3,-1)))

                # define two cameras
                cam_1,cam_2 = self.cameras[c1], self.cameras[c2]

                # error from Camera 1
                inter,idx1,idx2 = np.intersect1d(X[0],Track[0][0],assume_unique=True,return_indices=True)
                X_temp = util.homogeneous(X[1:,idx1])
                x = Track[0][1:,idx2]
                x_repro = cam_1.projectPoint(X_temp)
                err_x1,err_y1 = abs(x[0]-x_repro[0]), abs(x[1]-x_repro[1])

                # error from Camera 2
                inter,idx1,idx2 = np.intersect1d(X[0],Track[1][0],assume_unique=True,return_indices=True)
                X_temp = util.homogeneous(X[1:,idx1])
                x = Track[1][1:,idx2]
                x_repro = cam_2.projectPoint(X_temp)
                err_x2,err_y2 = abs(x[0]-x_repro[0]), abs(x[1]-x_repro[1])

                error = np.concatenate((err_x1,err_y1,err_x2,err_y2))

                # Make sure the error vector has a fixed size
                if len(err_x1) < len(self.traj[0]):
                    error = np.concatenate((error,np.zeros(2*(len(self.traj[0])-len(err_x1)))))
                if len(err_x2) < len(self.traj[0]):
                    error = np.concatenate((error,np.zeros(2*(len(self.traj[0])-len(err_x2)))))

                return error
            
            # before
            c1 = 0
            model = np.array([self.beta[0,c2]])
            model = np.concatenate((model,self.cameras[c1].P2vector_9(),self.cameras[c2].P2vector_9()))
            model = np.concatenate((model,np.ravel(self.traj[1:])))

            # during
            fn = lambda x: error_fn(x)
            res = least_squares(fn,model)

            # after
            self.beta[0,c2] = res.x[0]
            self.cameras[c1].vector2P_9(res.x[1:10])
            self.cameras[c2].vector2P_9(res.x[10:19])
            self.set_tracks()
            self.traj[1:] = res.x[19:].reshape(3,-1)
            
            return res


        def optimize_two_KRtd(self,c2):
            '''
            This function assumes that the first camera is the one started first, c2 is the index for the second camera

            K, R, t and distortion are all optimized
            '''

            def error_fn(model):

                self.beta[0,c2] = model[0]
                self.cameras[c1].vector2P_9(model[1:10])
                self.cameras[c2].vector2P_9(model[12:21])
                self.cameras[c1].d = model[10:12]
                self.cameras[c2].d = model[21:23]

                self.undistort_detections()
                self.set_tracks()

                Track = [self.tracks[c1],self.tracks[c2]]
                X = np.vstack((self.traj[0], model[23:].reshape(3,-1)))

                # define two cameras
                cam_1,cam_2 = self.cameras[c1], self.cameras[c2]

                # error from Camera 1
                inter,idx1,idx2 = np.intersect1d(X[0],Track[0][0],assume_unique=True,return_indices=True)
                X_temp = util.homogeneous(X[1:,idx1])
                x = Track[0][1:,idx2]
                x_repro = cam_1.projectPoint(X_temp)
                err_x1,err_y1 = abs(x[0]-x_repro[0]), abs(x[1]-x_repro[1])

                # error from Camera 2
                inter,idx1,idx2 = np.intersect1d(X[0],Track[1][0],assume_unique=True,return_indices=True)
                X_temp = util.homogeneous(X[1:,idx1])
                x = Track[1][1:,idx2]
                x_repro = cam_2.projectPoint(X_temp)
                err_x2,err_y2 = abs(x[0]-x_repro[0]), abs(x[1]-x_repro[1])

                error = np.concatenate((err_x1,err_y1,err_x2,err_y2))

                # Make sure the error vector has a fixed size
                if len(err_x1) < len(self.traj[0]):
                    error = np.concatenate((error,np.zeros(2*(len(self.traj[0])-len(err_x1)))))
                if len(err_x2) < len(self.traj[0]):
                    error = np.concatenate((error,np.zeros(2*(len(self.traj[0])-len(err_x2)))))

                return error
            
            # before
            c1 = 0
            model = np.array([self.beta[0,c2]])
            model = np.concatenate((model,self.cameras[c1].P2vector_9(),self.cameras[c1].d))
            model = np.concatenate((model,self.cameras[c2].P2vector_9(),self.cameras[c2].d))
            model = np.concatenate((model,np.ravel(self.traj[1:])))

            # during
            fn = lambda x: error_fn(x)
            res = least_squares(fn,model)

            # after
            self.beta[0,c2] = res.x[0]
            self.cameras[c1].vector2P_9(res.x[1:10])
            self.cameras[c1].d = res.x[10:12]
            self.cameras[c2].vector2P_9(res.x[12:21])
            self.cameras[c2].d = res.x[21:23]

            self.undistort_detections()
            self.set_tracks()

            self.traj[1:] = res.x[23:].reshape(3,-1)
            
            return res


        def optimize_all_Rt(self):
            
            def error_fn(model):
                self.beta[0,1:]  = model[:self.numCam-1]
                for i in range(self.numCam):
                    self.cameras[i].vector2Rt(model[self.numCam-1+i*6 : self.numCam-1+(i+1)*6])
                self.set_tracks()

                Track = self.tracks
                X = np.vstack((self.traj[0], model[self.numCam-1+self.numCam*6:].reshape(3,-1)))

                error = np.array([])
                for i in range(self.numCam):
                    cam_temp = self.cameras[i]

                    inter,idx1,idx2 = np.intersect1d(X[0],Track[i][0],assume_unique=True,return_indices=True)
                    X_temp = util.homogeneous(X[1:,idx1])
                    x = Track[i][1:,idx2]
                    x_repro = cam_temp.projectPoint(X_temp)
                    err_x,err_y = abs(x[0]-x_repro[0]), abs(x[1]-x_repro[1])

                    error = np.concatenate((error,err_x,err_y))
                    if len(err_x) < len(self.traj[0]):
                        error = np.concatenate((error,np.zeros(2*(len(self.traj[0])-len(err_x)))))

                return error
            
            # before
            model = self.beta[0,1:]
            for i in range(self.numCam):
                model = np.concatenate((model,self.cameras[i].Rt2vector()))
            model = np.concatenate((model,np.ravel(self.traj[1:])))

            # during
            fn = lambda x: error_fn(x)
            res = least_squares(fn,model)

            # after
            self.beta[0,1:]  = res.x[:self.numCam-1]
            for i in range(self.numCam):
                self.cameras[i].vector2Rt(res.x[self.numCam-1+i*6 : self.numCam-1+(i+1)*6])
            self.set_tracks()
            self.traj[1:] = res.x[self.numCam-1+self.numCam*6:].reshape(3,-1)

            return res


        def optimize_all_KRt(self):
            
            def error_fn(model):

                self.beta[0,1:]  = model[:self.numCam-1]
                for i in range(self.numCam):
                    self.cameras[i].vector2P_9(model[self.numCam-1+i*9 : self.numCam-1+(i+1)*9])
                self.set_tracks()

                Track = self.tracks
                X = np.vstack((self.traj[0], model[self.numCam-1+self.numCam*9:].reshape(3,-1)))

                error = np.array([])
                for i in range(self.numCam):
                    cam_temp = self.cameras[i]

                    inter,idx1,idx2 = np.intersect1d(X[0],Track[i][0],assume_unique=True,return_indices=True)
                    X_temp = util.homogeneous(X[1:,idx1])
                    x = Track[i][1:,idx2]
                    x_repro = cam_temp.projectPoint(X_temp)
                    err_x,err_y = abs(x[0]-x_repro[0]), abs(x[1]-x_repro[1])

                    error = np.concatenate((error,err_x,err_y))
                    if len(err_x) < len(self.traj[0]):
                        error = np.concatenate((error,np.zeros(2*(len(self.traj[0])-len(err_x)))))

                return error
            
            # before
            model = self.beta[0,1:]
            for i in range(self.numCam):
                model = np.concatenate((model,self.cameras[i].P2vector_9()))
            model = np.concatenate((model,np.ravel(self.traj[1:])))

            # during
            fn = lambda x: error_fn(x)
            res = least_squares(fn,model)

            # after
            self.beta[0,1:]  = res.x[:self.numCam-1]
            for i in range(self.numCam):
                self.cameras[i].vector2P_9(res.x[self.numCam-1+i*9 : self.numCam-1+(i+1)*9])
            self.set_tracks()
            self.traj[1:] = res.x[self.numCam-1+self.numCam*9:].reshape(3,-1)

            return res


        def optimize_all_KRtd(self):
            
            def error_fn(model):

                self.beta[0,1:]  = model[:self.numCam-1]
                for i in range(self.numCam):
                    self.cameras[i].vector2P_9(model[self.numCam-1+i*11 : self.numCam-1+i*11+9])
                    self.cameras[i].d = model[self.numCam-1+i*11+9 : self.numCam-1+i*11+11]
                self.undistort_detections()
                self.set_tracks()

                Track = self.tracks
                X = np.vstack((self.traj[0], model[self.numCam-1+self.numCam*11:].reshape(3,-1)))

                error = np.array([])
                for i in range(self.numCam):
                    cam_temp = self.cameras[i]

                    inter,idx1,idx2 = np.intersect1d(X[0],Track[i][0],assume_unique=True,return_indices=True)
                    X_temp = util.homogeneous(X[1:,idx1])
                    x = Track[i][1:,idx2]
                    x_repro = cam_temp.projectPoint(X_temp)
                    err_x,err_y = abs(x[0]-x_repro[0]), abs(x[1]-x_repro[1])

                    error = np.concatenate((error,err_x,err_y))
                    if len(err_x) < len(self.traj[0]):
                        error = np.concatenate((error,np.zeros(2*(len(self.traj[0])-len(err_x)))))

                return error
            
            # before
            model = self.beta[0,1:]
            for i in range(self.numCam):
                model = np.concatenate((model,self.cameras[i].P2vector_9()))
                model = np.concatenate((model,self.cameras[i].d))
            model = np.concatenate((model,np.ravel(self.traj[1:])))

            # during
            fn = lambda x: error_fn(x)
            res = least_squares(fn,model)

            # after
            self.beta[0,1:]  = res.x[:self.numCam-1]
            for i in range(self.numCam):
                self.cameras[i].vector2P_9(res.x[self.numCam-1+i*11 : self.numCam-1+i*11+9])
                self.cameras[i].d = res.x[self.numCam-1+i*11+9 : self.numCam-1+i*11+11]
            self.undistort_detections()
            self.set_tracks()
            self.traj[1:] = res.x[self.numCam-1+self.numCam*11:].reshape(3,-1)

            return res

                
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
        elif n==9:
            return np.concatenate((k,r,self.t))
        elif n==11:
            return np.concatenate((k,r,self.t,self.d))


    def vector2P(self,vector,n=6):
        '''
        Convert a vector into camera parameters
        '''

        if n==6:
            self.R = cv2.Rodrigues(vector[:3])[0]
            self.t = vector[3:6]
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

        self.compose()
        return self.P


    def discard_fn(self):
        '''
        Functions that are not needed (for now) are stored here
        '''

        def P2vector_10(self):
            k = np.array([self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]])
            r = cv2.Rodrigues(self.R)[0]
            return np.concatenate((k,r.reshape(-1,),self.t))

            # r1,r2,r3 = util.rotation_decompose(self.R)
            # return np.concatenate((k,np.array([r1,r2,r3]),self.t))

        
        def P2vector_9(self):
            k = np.array([np.mean(self.K[0,0]+self.K[1,1]), self.K[0,2], self.K[1,2]])
            r = cv2.Rodrigues(self.R)[0]
            return np.concatenate((k,r.reshape(-1,),self.t))


        def vector2P_10(self,vector):
            self.K = np.diag((1,1,1)).astype(float)
            self.K[0,0], self.K[1,1] = vector[0], vector[1]
            self.K[:2,-1] = vector[2:4]
            self.R = cv2.Rodrigues(vector[4:7])[0]
            self.t = vector[7:]

            self.compose()
            return self.P


        def vector2P_9(self,vector):
            self.K = np.diag((1,1,1)).astype(float)
            self.K[0,0], self.K[1,1] = vector[0], vector[0]
            self.K[:2,-1] = vector[1:3]
            self.R = cv2.Rodrigues(vector[3:6])[0]
            self.t = vector[6:]

            self.compose()
            return self.P

        
        def Rt2vector(self):
            r = cv2.Rodrigues(self.R)[0]
            return np.concatenate((r.reshape(-1,),self.t))


        def vector2Rt(self,vector):
            self.R = cv2.Rodrigues(vector[:3])[0]
            self.t = vector[3:]

            self.compose()
            return self.P


class trackedObject:
    """
    Class that describes individual objects to be tracked

    This class contains information about moving objects in the images, e.g. their 2D tracks, 3D trajectories etc.

    Members
    -------
    imageTracks : list of lists of lists of int
        All detections of given object in all cameras. The object can be tracked in more than one camera and have more than one continuous trajectory in each camera (when the object goes out of the cameras FOV or is occluded). The following should give a third continuous track in the second camera: 
            imageTracks[1][2]
        which itself is a list of tuples (frameId,x,y) where frameId is the id of the frame in which the object was detected at image coordinates x,y.
    """

    def __init__(self,num_cam):
        self.num_cam = num_cam
        self.imageTracks = [[] for j in range(num_cam)]


def detect_undistort(d,cameras):
    '''Use Opencv undistpoints'''

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

    Track_all = [[] for i in range(len(detections))]
    idx_min = 0
    for i in range(len(detections)):
        track = copy.deepcopy(detections[i])
        track[0] -= track[0,0]

        if i == 0:
            Track_all[i] = track
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


def jac_twoCam(N,K=False):

    jac_X = np.zeros((4*N,3*N))
    for i in range(N):
        jac_X[(i,i+N,i+2*N,i+3*N),i*3:(i+1)*3] = 1

    if K:
        cam_param = 9
    else:
        cam_param = 6

    jac_cam = np.zeros((4*N,2*cam_param))
    jac_cam[:2*N,:cam_param] = 1
    jac_cam[2*N:,cam_param:] = 1

    return np.hstack((jac_cam,jac_X))


def optimze_two(c1,c2,x1,x2,X,include_K=False):

    def error_fn(model):
        cam_param = int((len(model)-x1.shape[1]*3)/2)
        c1.vector2P(model[:cam_param],n=n_cam)
        c2.vector2P(model[cam_param:2*cam_param],n=n_cam)
        X = model[2*cam_param:].reshape(-1,3).T

        x_c1 = c1.projectPoint(util.homogeneous(X))
        x_c2 = c2.projectPoint(util.homogeneous(X))

        return np.concatenate((abs(x1[0]-x_c1[0]),abs(x1[1]-x_c1[1]),abs(x2[0]-x_c2[0]),abs(x2[1]-x_c2[1])))
    
    # before
    if include_K: 
        n_cam = 9 
    else: 
        n_cam = 6
    model = np.concatenate((c1.P2vector(n=n_cam),c2.P2vector(n=n_cam),np.ravel(X.T)))

    # During
    bound_low  = np.ones(len(model))*-np.inf
    bound_high = np.ones(len(model))*np.inf
    A = jac_twoCam(x1.shape[1],K=include_K)

    fn = lambda x: error_fn(x)
    res = least_squares(fn,model,bounds=(bound_low,bound_high),jac_sparsity=A)

    # After
    cam_param = int((len(model)-x1.shape[1]*3)/2)
    c1.vector2P(res.x[:cam_param],n=n_cam)
    c2.vector2P(res.x[cam_param:2*cam_param],n=n_cam)
    X = res.x[2*cam_param:].reshape(-1,3).T

    return res,[c1,c2,X]


def jac_allCam(v,n):
    num_cam, num_point = v.shape[0], v.shape[1]

    jac_X = np.zeros((num_point*2,num_point*3))
    for i in range(num_point):
        jac_X[(i,i+num_point),i*3:(i+1)*3] = 1
    
    jac = np.empty(())
    for i in range(num_cam):
        jac_cam_i = np.zeros((num_point*2,num_cam*n))
        jac_cam_i[:,i*n:(i+1)*n] = 1

        jac_i = np.hstack((jac_cam_i,jac_X))
        for j in range(num_point):
            if v[i,j] == 0:
                jac_i[(j,j+num_point),:] = 0

        if i == 0:
            jac = jac_i
        else:
            jac = np.vstack((jac,jac_i))

    return jac


def optimize_all(cams,tracks,traj,v,include_K=False):

    def error_fn(model):

        error = np.array([])
        traj[1:] = model[num_Cam*n_cam:].reshape(-1,3).T
        for i in range(num_Cam):
            cams[i].vector2P(model[i*n_cam:(i+1)*n_cam],n=n_cam)
            
            inter,idx1,idx2 = np.intersect1d(traj[0],tracks[i][0],assume_unique=True,return_indices=True)
            X_temp = util.homogeneous(traj[1:,idx1])
            x = tracks[i][1:,idx2]
            x_repro = cams[i].projectPoint(X_temp)

            error_temp = np.array([abs(x_repro[0]-x[0]),abs(x_repro[1]-x[1])])
            error_cam_i = np.zeros(num_Point*2)
            for j in range(len(inter)):
                error_cam_i[int(idx1[j])],error_cam_i[int(idx1[j])+num_Point] = error_temp[0,j], error_temp[1,j]

            error = np.concatenate((error,error_cam_i))
        return error
    
    # before
    if include_K: 
        n_cam = 9 
    else: 
        n_cam = 6
    num_Cam, num_Point = len(cams), traj.shape[1]

    model = np.array([])
    for i in cams:
        model = np.concatenate((model,i.P2vector(n=n_cam)))
    model = np.concatenate((model,np.ravel(traj[1:].T)))

    # During
    A = jac_allCam(v,n_cam)
    fn = lambda x: error_fn(x)
    res = least_squares(fn,model,jac_sparsity=A)

    # After
    for i in range(num_Cam):
        cams[i].vector2P(res.x[i*n_cam:(i+1)*n_cam],n=n_cam)
    traj[1:] = res.x[num_Cam*n_cam:].reshape(-1,3).T

    return res,[cams,traj]


def test_two():
    pass


def test_all():
    pass


if __name__ == "__main__":

    '''
    Parameters:

    1. detection in int or float
    2. undistort images or not
    3. Using spline or not
    4. triangulate all points or not
    5. loss function mean or not

    '''

    # Load previous scene..
    # with open('./data/jobs/EF_KRt/optimization_before_E_KRt.pkl', 'rb') as file:
    #     f1 = pickle.load(file)
    # with open('./data/jobs/EF_KRt/optimization_after_E_KRt.pkl', 'rb') as file:
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
    use_spline = False

    # Compute beta for every pair of cameras
    flight.beta = np.array([[0,-81.74,-31.73],[81.74,0,52.25],[31.73,-52.25,0]])
    # flight.compute_beta(spline=use_spline,threshold_error=2)
    # print('\n',flight.beta,'\n')

    # Sort detections in temporal order
    flight.set_sequence()

    # create tracks according to beta
    flight.set_tracks(spline=use_spline)

    # Initialize the first 3D trajectory
    use_F = False
    include_K = True
    if use_F:
        error = 25
    else:
        error = 0.006
    print('\nCurrently using Fundamental matrix:{}, threshold:{}, K is optimized:{}\n'.format(use_F,error,include_K))
    idx1, idx2 = flight.init_traj(error=error,F=use_F,inlier_only=True)



    '''----------------Optimization----------------'''
    start=datetime.now()

    flight.error_cam(0)
    flight.error_cam(2)

    '''Optimizate two'''
    res, model = optimze_two(flight.cameras[0],flight.cameras[2],flight.tracks[0][1:,idx1],
                           flight.tracks[2][1:,idx2],flight.traj[1:],include_K=include_K)
    flight.cameras[0],flight.cameras[2],flight.traj[1:] = model[0], model[1], model[2]

    print('\nTime: ',datetime.now()-start)

    # Get camera poses by solving PnP
    flight.get_camera_pose(flight.sequence[2],error=10)

    # Check reprojection error
    flight.error_cam(0)
    flight.error_cam(1)
    flight.error_cam(2)

    # Triangulate more points if possible
    flight.triangulate_traj(0,2)
    flight.triangulate_traj(1,2)

    flight.error_cam(1)

    # Define visibility
    flight.set_visibility()

    '''Optimizate all'''
    flight_before = copy.deepcopy(flight)

    with open('./data/optimization_before_E_KRt.pkl', 'wb') as f:
        pickle.dump(flight, f)

    res, model = optimize_all(flight.cameras,flight.tracks,flight.traj,flight.visible,include_K=include_K)
    flight.cameras, flight.traj = model[0], model[1]

    # Check reprojection error
    flight.error_cam(0)
    flight.error_cam(1)
    flight.error_cam(2)
    
    # # Visualize the 3D trajectory
    # vis.show_trajectory_3D(flight.traj[1:],line=False)
    # vis.show_trajectory_2D(flight.tracks[0][1:],flight.cameras[0].projectPoint(util.homogeneous(flight.traj[1:])))
    # vis.show_trajectory_2D(flight.tracks[2][1:],flight.cameras[2].projectPoint(util.homogeneous(flight.traj[1:])))

    with open('./data/optimization_after_E_KRt.pkl', 'wb') as f:
        pickle.dump(flight, f)

    print('\nTime: ',datetime.now()-start)

    print('\nFinished')