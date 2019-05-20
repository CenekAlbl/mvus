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


    def init_beta(self):
        '''Initialize the beta vector according to numCam'''
        self.beta = np.zeros((self.numCam, self.numCam))

    
    def compute_beta(self,d_min=-6,d_max=6,threshold_1=10,threshold_2=2,threshold_error=3):
        while True:
            beta_error = 0
            for i in range(self.numCam-1):
                for j in range(i+1,self.numCam):
                    d1 = util.homogeneous(self.detections[i][1:])
                    d2 = util.homogeneous(self.detections[j][1:])
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

    
    def set_sequence(self,*arg):
        if len(arg):
            self.sequence = arg[0]
        else:
            self.sequence = self.beta[0].argsort()[::-1]


    def set_tracks(self,auto=True,*beta):

        if not len(beta):
            beta = self.beta[0]
        else:
            beta = beta[0]

        Track_all = detect_to_track(self.detections,beta)

        if auto:
            self.tracks = Track_all
            return Track_all
        else:
            return Track_all
    

    def init_traj(self,error=10,inlier_only=False):
        '''
        Select the first two tracks in the sequence, compute fundamental matrix, triangulate points
        '''

        t1, t2 = self.sequence[0], self.sequence[1]
        inter,idx1,idx2 = np.intersect1d(self.tracks[t1][0],self.tracks[t2][0],assume_unique=True,return_indices=True)
        track_1, track_2 = self.tracks[t1][:,idx1], self.tracks[t2][:,idx2]

        F,inlier = ep.computeFundamentalMat(track_1[1:],track_2[1:],error=error)
        if not inlier_only:
            inlier = np.ones(len(inlier))
        x1, x2 = util.homogeneous(track_1[1:,inlier==1]), util.homogeneous(track_2[1:,inlier==1])
        K1, K2 = self.cameras[t1].K, self.cameras[t2].K
        E = np.dot(np.dot(K2.T,F),K1)

        # X, P = ep.triangulate_from_E_old(E,K1,K2,x1,x2)
        X, P = ep.triangulate_from_E(E,K1,K2,x1,x2)
        self.traj = np.vstack((inter[inlier==1],X[:-1]))

    
    def get_camera_pose(self,id_cam,thres=0.01):
        inter,idx1,idx2 = np.intersect1d(self.traj[0],self.tracks[id_cam][0],assume_unique=True,return_indices=True)
        X = util.homogeneous(self.traj[1:,idx1])
        x = util.homogeneous(self.tracks[id_cam][1:,idx2])
        x_n = np.dot(np.linalg.inv(self.cameras[id_cam].K), x)

        P,inlier = ep.solve_PnP_Ransac(x_n,X,threshold=thres)
        Cam_temp = Camera(P=np.dot(self.cameras[id_cam].K, P))
        # P,inlier = ep.solve_PnP_Ransac(x,X,threshold=8)
        # Cam_temp = Camera(P=P)

        x_cal = Cam_temp.projectPoint(X)
        error = ep.reprojection_error(x,x_cal)
        print("Mean reprojection error: ",np.mean(error))

        self.cameras[id_cam] = Cam_temp
        self.cameras[id_cam].decompose()

    
    def error_cam(self,id_cam):
        inter,idx1,idx2 = np.intersect1d(self.traj[0],self.tracks[id_cam][0],assume_unique=True,return_indices=True)
        X = util.homogeneous(self.traj[1:,idx1])
        x = util.homogeneous(self.tracks[id_cam][1:,idx2])
        
        x_cal = self.cameras[id_cam].projectPoint(X)
        error = ep.reprojection_error(x,x_cal)
        return error


    def optimize(self):
        
        def error_fn(model,data,**param):
            numCam = param.get('numCam')
            ind_X = param.get('ind_X')

            Track = detect_to_track(data,np.concatenate(([0],model[:numCam-1])))
            X = np.vstack((ind_X, model[numCam*12-1:].reshape(3,-1)))

            error = np.array([])
            for i in range(numCam):
                cam_temp = Camera()
                cam_temp.vector2P(model[numCam-1+i*11:numCam-1+i*11+11])

                inter,idx1,idx2 = np.intersect1d(X[0],Track[i][0],assume_unique=True,return_indices=True)
                X_temp = util.homogeneous(X[1:,idx1])
                x = util.homogeneous(Track[i][1:,idx2])
                
                x_cal = cam_temp.projectPoint(X_temp)
                err = ep.reprojection_error(x,x_cal)
                error = np.concatenate((error,err))

            return error
        
        # before
        model = self.beta[0,1:]
        for i in range(self.numCam):
            model = np.concatenate((model,self.cameras[i].P2vector()))
        model = np.concatenate((model,np.ravel(self.traj[1:])))

        # during
        fn = lambda x: error_fn(x,self.detections,numCam=self.numCam,ind_X=self.traj[0])
        res = least_squares(fn,model)

        # after
        self.beta[0]  = np.concatenate(([0],res.x[:self.numCam-1]))
        self.tracks   = detect_to_track(self.detections,self.beta[0])
        self.traj[1:] = res.x[self.numCam*12-1:].reshape(3,-1)
        for i in range(self.numCam):
            self.cameras[i].vector2P(res.x[self.numCam-1+i*11 : self.numCam-1+i*11+11])
            err = self.error_cam(i)
            print("Mean reprojection error after optimization: ",np.mean(err))
        
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


    def P2vector(self):
        K = np.concatenate((self.K[0],self.K[1,1:]))
        r1,r2,r3 = util.rotation_decompose(self.R)
        
        return np.concatenate((K,np.array([r1,r2,r3]),self.t))

    
    def vector2P(self,vector):
        self.K = np.diag((1,1,1)).astype(float)
        self.K[0] = vector[:3]
        self.K[1,1:] = vector[3:5]
        self.R = util.rotation(vector[5],vector[6],vector[7])
        self.t = vector[8:]

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


if __name__ == "__main__":

    # Load previous scene..
    # with open('./data/optimization_before.pkl', 'rb') as file:
    #     before = pickle.load(file)
    # with open('./data/optimization_after.pkl', 'rb') as file:
    #     after = pickle.load(file)

    # New scene
    start=datetime.now()
    
    # Load camara intrinsic and radial distortions
    intrin_1 = scio.loadmat('./data/calibration/first_flight/gopro/calibration_narrow.mat')
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
    # flight.addDetection(detect_1.T.astype(float), detect_2.T.astype(float), detect_3.T.astype(float))
    flight.addDetection(detect_1.T, detect_2.T, detect_3.T)
    flight.init_beta()

    # Undistort images according to calibration
    for i in range(flight.numCam):
        detection_norm   = np.dot(np.linalg.inv(flight.cameras[i].K), util.homogeneous(flight.detections[i][1:]))
        detection_unnorm = np.dot(flight.cameras[i].K, ep.undistort(detection_norm,flight.cameras[i].d))
        flight.detections[i][1:] = detection_unnorm[:2]

    # Compute beta for every pair of cameras
    flight.beta = np.array([[0,-81.74,-31.73],[81.74,0,52.25],[31,73.25,0]])
    # flight.compute_beta()

    # Sort detections in temporal order
    flight.set_sequence()

    # create tracks according to beta
    flight.set_tracks()

    # Initialize the first 3D trajectory
    flight.init_traj(error=3,inlier_only=True)

    # Get camera poses by solving PnP
    flight.get_camera_pose(0)
    flight.get_camera_pose(1)
    flight.get_camera_pose(2)

    # Visualize the 3D trajectory
    # vis.show_trajectory_3D(flight.traj[1:])

    # Save scene before optimization
    with open('./data/optimization_before.pkl', 'wb') as f:
        pickle.dump(flight, f)
    
    # Optimization
    start=datetime.now()

    test = 1
    for i in range(flight.numCam):
        test *= np.mean(flight.error_cam(i)) < 100

    if test:
        result = flight.optimize()
        print('Results of beta: \n', result.x[:2])
        print('Results of camaras: \n', result.x[2:35])
    else:
        print('Reprojection error too large!')

    print('\nTime: ',datetime.now()-start)

    # Save scene after optimization
    with open('./data/optimization_after.pkl', 'wb') as f:
        pickle.dump(flight, f)

    # # model
    # model = flight.beta[0,1:]
    # for i in range(flight.numCam):
    #     model = np.concatenate((model,flight.cameras[i].P2vector()))
    # model = np.concatenate((model,np.ravel(flight.traj[1:])))

    # # data
    # Track = flight.set_tracks(False,np.concatenate(([0],model[:flight.numCam-1])))

    # X = np.vstack((flight.traj[0],model[flight.numCam*12-1:].reshape(3,-1)))

    # for i in range(flight.numCam):
    #     cam_temp = Camera()
    #     cam_temp.vector2P(model[flight.numCam-1+i*11:flight.numCam-1+i*11+11])

    #     # print(util.rotation_decompose(flight.cameras[i].R))
    #     # print(util.rotation_decompose(cam_temp.R))
    #     # print(flight.cameras[i].R-cam_temp.R)

    #     inter,idx1,idx2 = np.intersect1d(X[0].astype(int),Track[i][0],assume_unique=True,return_indices=True)
    #     X_temp = util.homogeneous(X[1:,idx1])
    #     x = util.homogeneous(Track[i][1:,idx2])
        
    #     x_cal = cam_temp.projectPoint(X_temp)
    #     error = ep.reprojection_error(x,x_cal)
    #     print("Mean reprojection error: ",np.mean(error))



    # # Triangulate more points
    # inter,idx1,idx2 = np.intersect1d(flight.tracks[t2][0],flight.tracks[t3][0],assume_unique=True,return_indices=True)
    # x1 = np.dot(np.linalg.inv(flight.cameras[t2].K), util.homogeneous(flight.tracks[t2][1:,idx1]))
    # x2 = np.dot(np.linalg.inv(flight.cameras[t3].K), util.homogeneous(flight.tracks[t3][1:,idx2]))
    # # x1 = util.homogeneous(flight.tracks[t2][1:,idx1])
    # # x2 = util.homogeneous(flight.tracks[t3][1:,idx2])
    # X_new = ep.triangulate_matlab(x1,x2,Camera_temp_2.P,Camera_temp_3.P)
    # traj_new = np.vstack((inter,X_new[:-1]))

    # vis.show_trajectory_3D(X_new,line=False)

    # idx_new = np.in1d(traj_new[0].astype(int),flight.traj[0].astype(int),invert=True)
    # flight.traj = np.hstack((flight.traj,traj_new[:,idx_new]))
    # flight.traj = flight.traj[:,np.argsort(flight.traj[0])]

    # vis.show_trajectory_3D(X_temp,flight.traj[1:],line=False)

    print('\nFinished')