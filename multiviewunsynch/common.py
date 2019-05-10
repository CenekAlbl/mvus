# Classes that are common to the entire project
import numpy as np
import util
import epipolar as ep
import synchronization
import scipy.io as scio
import pickle

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

    
    def init_track(self):
        '''Initialize the tracks identical to detections'''
        self.tracks = self.detections


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
        self.c = kwargs.get('c')


    def projectPoint(self,X):
        x = np.dot(self.P,X)
        x /= x[2]
        return x


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

        return self.K, self.R, self.t


    def center(self):
        if self.c is not None:
            return self.c
        else:
            self.decompose()
            self.c = -np.dot(self.R.T,self.t)
            return self.c


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


if __name__ == "__main__":
    
    # Load camara intrinsic
    intrin_1 = scio.loadmat('./data/calibration/first_flight/gopro/calibration_narrow.mat')
    intrin_2 = scio.loadmat('./data/calibration/first_flight/phone_1/calibration.mat')
    intrin_3 = scio.loadmat('./data/calibration/first_flight/phone_2/calibration.mat')

    K1, K2, K3 = intrin_1['intrinsic'], intrin_2['intrinsic'], intrin_3['intrinsic']
    cameras = [Camera(K=K1), Camera(K=K2), Camera(K=K3)]

    # Load detections
    detect_1 = np.loadtxt('./data/video_1_output.txt',skiprows=1,dtype=np.int32)
    detect_2 = np.loadtxt('./data/video_2_output.txt',skiprows=1,dtype=np.int32)
    detect_3 = np.loadtxt('./data/video_3_output.txt',skiprows=1,dtype=np.int32)

    # Create a scene
    flight = Scene()
    flight.addCamera(*cameras)
    flight.addDetection(detect_1.T, detect_2.T, detect_3.T)
    flight.init_beta()
    flight.init_track()

    # Compute Beta for the first two detections
    beta_error = 0
    for i in range(flight.numCam-1):
        for j in range(i+1,flight.numCam):
            d1 = util.homogeneous(flight.detections[i][1:])
            d2 = util.homogeneous(flight.detections[j][1:])
            numPoints = min(d1.shape[1], d2.shape[1])

            param = {'k':1, 's':0}
            flight.beta[i,j], F, inliers = synchronization.search_sync(d1[:,:numPoints], d2[:,:numPoints], param=param, d_min=-6, d_max=6, threshold2=5)
            flight.beta[j,i] = flight.beta[i,j]
        beta_error += flight.beta[i,i+1]
    beta_error -= flight.beta[0,-1]

    print('\nBeta error: {}\n'.format(beta_error))
    print(flight.beta)

    print('\nFinished')