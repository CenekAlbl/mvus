# Classes that are common to the entire project
import numpy as np

class Scene:
    """ 
    Class that contains everything that we know about the scene, both measured and calculated data

    This class will contain all data and will be passed around to be changed and used by most methods. 

    Members
    -------
    cameras : list of elements of class Camera 
    trackedObjects : list of elements of class TrackedObject
    """
    def __init__(self):
        """ 
        Default constructor, creates a Scene with no data
        """
        self.cameras = []
        self.trackedObjects = []
        pass

    def addCamera(self,camera):
        """Adds a camera to the scene
        
        Adds a camera to the scene, first it checks whether the passed object is an instance of Camera

        Parameters
        ----------
        camera : Camera
            the camera to be added
        """
        assert type(camera) is Camera, "camera is not an instance of Camera"
        self.cameras.append(camera)


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

    Methods:
    projectPoint 

    """

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