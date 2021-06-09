# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Classes that are common to the entire project
import numpy as np
from scipy.optimize._lsq.least_squares import FROM_MINPACK_TO_COMMON
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
        self.beta_after_Fbeta = []
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

        # save data regarding static part
        # the reconstructed 3D points for the static scene
        self.static = np.empty([3, 0])
        # the inlier mask of the reconstructed 3D static points (to keep a track of the inliers )
        self.inlier_mask = np.empty([0])
        # the ground truth static 3d points
        self.gt_static = None
        # the dictionary stores the matching result
        self.feature_dict = {}  
        # feature_dict[cam_id] = np.array((n_cam, n_kp)) feature_dict[cam1][cam2]: values=indices of matched features in cam2, col=indices of matched features in cam1
        # e.g. feature_dict[0] = [[-1,-1,-1,-1,-1,-1],
        #                         [-1,-1, 2, 1,-1, 3],
        #                         [ 3,-1,-1,-1, 0, -1]]
        #      feature_dict[1] = [[-1, 3, 2, 5,-1,-1],
        #                         [-1,-1,-1,-1,-1,-1],
        #                         [ 1,-1, 4, 0, 2, 3]]
        #      feature_dict[2] = [[ 4,-1,-1, 0,-1,-1],
        #                         [ 3, 0, 4, 5, 2,-1],
        #                         [-1,-1,-1,-1,-1,-1]]

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
            detect = self.cameras[i].undist_point(self.detections[i][1:], self.settings['undist_method']) if self.settings['undist_points'] else self.detections[i][1:]
            self.detections_global[i] = np.vstack((timestamp, detect))

            if motion_prior:
                if (self.global_traj[1] == i).any():
                    # Update glob_traj timestamps for current camera
                    temp_glob_traj = self.global_traj[:,self.global_traj[1] == i]
                    # Save traj. point locations in global_traj before update
                    temp_glob_traj_mask = np.where(self.global_traj[1] == i)
                    # Select global det. points for current camera
                    temp_glob_det = self.global_detections[:,self.global_detections[0] == i]
                    # Save traj. point locations in global_traj
                    temp_glob_det_mask = np.where(self.global_detections[0] == i)
                    # Save camera detections that are used within the global traj. 
                    _,temp_glob_traj_idx,temp_glob_traj_det_idx = np.intersect1d(temp_glob_traj[2],self.detections[i][0],return_indices=True,assume_unique=True)
                    # Save camera detections that are used within global detections. 
                    _,temp_glob_det_idx,temp_det_idx = np.intersect1d(temp_glob_det[1],self.detections[i][0],return_indices=True,assume_unique=True)  
                    
                    assert np.shape(temp_glob_traj_mask)[1] == np.shape(temp_glob_traj_det_idx)[0],'# of 3D points must equal # of detections'
                    assert np.shape(temp_glob_det_mask)[1] == np.shape(self.detections_global[i][0])[0],'# of 2D points must equal # of selected global detections'
                    
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


    def init_traj(self,error=10,inlier_only=False, debug=False):
        '''
        Function:
            Select the first two cams in the sequence, compute fundamental matrix, triangulate points
        Input:
            debug = True -- use the ground truth matches for the static part;
                    False -- use the extracted features
        '''

        self.select_most_overlap(init=True)

        t1, t2 = self.sequence[0], self.sequence[1]
        K1, K2 = self.cameras[t1].K, self.cameras[t2].K

        # Find correspondences
        if self.cameras[t1].fps > self.cameras[t2].fps:
            d1, d2 = util.match_overlap(self.detections_global[t1], self.detections_global[t2])
        else:
            d2, d1 = util.match_overlap(self.detections_global[t2], self.detections_global[t1])
        
        # draw matches between dections
        if self.settings['undist_points']:
            # the background images are the original ones and are not undistorted, the detections need to be distorted
            d1_dist = self.cameras[t1].dist_point2d(d1[1:], method=self.settings['undist_method'])
            d2_dist = self.cameras[t2].dist_point2d(d2[1:], method=self.settings['undist_method'])

            vis.draw_detection_matches(self.cameras[t1].img, np.vstack([d1[0], d1_dist]), self.cameras[t2].img, np.vstack([d2[0],d2_dist]))
        else:
            vis.draw_detection_matches(self.cameras[t1].img, d1, self.cameras[t2].img, d2)
        
        # add the static part
        if 'include_static' in self.settings.keys() and self.settings['include_static']:
            if debug:
                # in debug, use static ground truth as 2d featues
                if self.settings['undist_points']:
                    # undistort the ground truth matches
                    pts1 = self.cameras[t1].undist_point(self.cameras[t1].gt_pts.T, self.settings['undist_method']).T
                    pts2 = self.cameras[t2].undist_point(self.cameras[t2].gt_pts.T, self.settings['undist_method']).T

                    # plot the ground truth matches
                    # FIXME: check pts_dist == gt_pts
                    pts1_dist = self.cameras[t1].dist_point2d(pts1.T, method=self.settings['undist_method'])
                    pts2_dist = self.cameras[t2].dist_point2d(pts2.T, method=self.settings['undist_method'])
                    # vis.draw_detection_matches(self.cameras[t1].img, np.vstack([np.zeros(pts1_dist.shape[1]),pts1_dist]), self.cameras[t2].img, np.vstack([np.zeros(pts2_dist.shape[1]),pts2_dist]))
                    vis.draw_detection_matches(self.cameras[t1].img, np.vstack([np.zeros(pts1.shape[0]),self.cameras[t1].gt_pts.T]), self.cameras[t2].img, np.vstack([np.zeros(pts2.shape[0]),self.cameras[t2].gt_pts.T]))

                else:
                    pts1 = self.cameras[t1].gt_pts
                    pts2 = self.cameras[t2].gt_pts
                
                    vis.draw_detection_matches(self.cameras[t1].img, np.vstack([np.zeros(pts1.shape[0]),pts1.T]), self.cameras[t2].img, np.vstack([np.zeros(pts2.shape[0]),pts2.T]))
            else:
                
                # sp1, sp2 = util.match_features(self.cameras[t1].img, self.cameras[t2].img, 'sift', 'bf', 0.7)

                # Match features
                pts1, pts2, matches, matchesMask = ep.matching_feature(self.cameras[t1].kp, self.cameras[t2].kp, self.cameras[t1].des, self.cameras[t2].des, ratio=0.8)
                # draw matches
                vis.draw_matches(self.cameras[t1].img, self.cameras[t1].kp, self.cameras[t2].img, self.cameras[t2].kp, matches, matchesMask)
                
                # undistort the matched keypoints
                if self.settings['undist_points']:
                    pts1 = self.cameras[t1].undist_point(np.array(pts1).T, self.settings['undist_method']).T
                    pts2 = self.cameras[t2].undist_point(np.array(pts2).T, self.settings['undist_method']).T
            
            # stack the static features with the detections and use them together for initial pose extraction
            fp1 = np.hstack([np.int32(pts1).T, d1[1:]])
            fp2 = np.hstack([np.int32(pts2).T, d2[1:]])

            X, P, inlier, mask = ep.epipolar_pipeline(fp1, fp2, K1, K2, error, inlier_only, self.cameras[t1].img, self.cameras[t2].img)
            
            # split into traj and static
            idx = np.where(inlier == 1)[0]
            idx_mask = idx[mask]
            inlier_static = idx_mask[idx_mask < len(pts1)]
            inlier_traj = idx_mask[idx_mask >= len(pts1)] - len(pts1)

            # save static part
            self.static = X[:-1, idx_mask < len(pts1)]
            self.inlier_mask = np.ones(self.static.shape[1])

            # get the matching indices
            if debug:
                # query_ids -- the index of the features in cam1
                query_ids = np.arange(len(pts1)) 
                # train_ids -- index of the features in cam2
                train_ids = np.arange(len(pts2))

                # initialize the matching result to be stored to the dict
                match_res1 = -np.ones((self.numCam, len(pts1)))
                match_res2 = -np.ones((self.numCam, len(pts2)))
            else:
                query_ids = np.array([m[0].queryIdx for m in matches])
                train_ids = np.array([m[0].trainIdx for m in matches])

                # draw matches
                matchesMask_inliers = np.zeros((len(matches), 2))
                match_ids = np.where(np.array(matchesMask)[:, 0] == 1)[0]
                match_ids_inliers = match_ids[inlier_static]
                matchesMask_inliers[match_ids_inliers] = [1, 0]

                vis.draw_matches(self.cameras[t1].img, self.cameras[t1].kp, self.cameras[t2].img, self.cameras[t2].kp, matches, matchesMask_inliers)

                query_ids = query_ids[match_ids]
                train_ids = train_ids[match_ids]

                # initialize the matching result to be stored to the dict
                match_res1 = -np.ones((self.numCam, len(self.cameras[t1].kp)))
                match_res2 = -np.ones((self.numCam, len(self.cameras[t2].kp)))
        
            # save static 2d to cameras
            self.cameras[t1].index_registered_2d = query_ids[inlier_static]
            self.cameras[t1].index_2d_3d = np.arange(self.static.shape[1])
            self.cameras[t2].index_registered_2d = train_ids[inlier_static]
            self.cameras[t2].index_2d_3d = np.arange(self.static.shape[1])

            match_res1[t2, query_ids] = train_ids
            match_res2[t1, train_ids] = query_ids

            self.feature_dict[t1] = match_res1
            self.feature_dict[t2] = match_res2

            # save trajectory
            self.traj = np.vstack((d1[0][inlier_traj], X[:-1, idx_mask >= len(pts1)]))
        
        # only uses the detections for pose estimation
        else:
            X, P, inlier, mask = ep.epipolar_pipeline(d1[1:], d2[1:], K1, K2, error, inlier_only, self.cameras[t1].img, self.cameras[t2].img)
            self.traj = np.vstack((d1[0][inlier==1][mask],X[:-1]))
        
        # # Compute fundamental matrix
        # F,inlier = ep.computeFundamentalMat(d1[1:],d2[1:],error=error)
        # E = np.dot(np.dot(K2.T,F),K1)

        # if not inlier_only:
        #     inlier = np.ones(len(inlier))
        # x1, x2 = util.homogeneous(d1[1:,inlier==1]), util.homogeneous(d2[1:,inlier==1])

        # # Find corrected corresponding points for optimal triangulation
        # N = d1[1:,inlier==1].shape[1]
        # pts1=d1[1:,inlier==1].T.reshape(1,-1,2)
        # pts2=d2[1:,inlier==1].T.reshape(1,-1,2)
        # m1,m2 = cv2.correctMatches(F,pts1,pts2)
        # x1,x2 = util.homogeneous(np.reshape(m1,(-1,2)).T), util.homogeneous(np.reshape(m2,(-1,2)).T)

        # mask = np.logical_not(np.isnan(x1[0]))
        # x1 = x1[:,mask]
        # x2 = x2[:,mask]

        # # Triangulte points
        # X, P = ep.triangulate_from_E(E,K1,K2,x1,x2)
        # self.traj = np.vstack((d1[0][inlier==1][mask],X[:-1]))

        # Assign the camera matrix for these two cameras
        self.cameras[t1].P = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
        self.cameras[t2].P = np.dot(K2,P)
        self.cameras[t1].decompose()
        self.cameras[t2].decompose()
    
    def init_traj_from_pose(self):
        t1, t2 = self.sequence[0], self.sequence[1]
        c1, c2 = -np.dot(self.cameras[t1].R.T, self.cameras[t1].t.reshape(-1,1)), -np.dot(self.cameras[t2].R.T, self.cameras[t2].t.reshape(-1,1))
        R12 = np.dot(self.cameras[t1].R.T, self.cameras[t2].R)
        t12 = -np.dot(R12, c2-c1).ravel()
        E = np.dot(self.cameras[t2].R, ep.skew(self.cameras[t2].t))
        E1 = np.dot(R12, ep.skew(t12))
        K1, K2 = self.cameras[t1].K, self.cameras[t2].K

        F = np.dot(np.linalg.inv(K2).T, np.dot(E, np.linalg.inv(K1)))

        # x1, x2 = util.homogeneous(self.cameras[t1].get_gt_pts()), util.homogeneous(self.cameras[t2].get_gt_pts())
        # l1 = F.dot(x1).T
        # # compute distance between l1 and x2
        # dist12 = np.dot(l1, x2)
        # dist12 /= np.sqrt(dist12[:,0]**2+dist12[:,1]**2).reshape(-1,1)
        # matched_ids1 = np.argmin(dist12, axis=0)
        # total_dist12 = np.mean(dist12[matched_ids1, np.arange(x2.shape[1])])

        # l2 = F.T.dot(x2).T
        # # compute distance between l1 and x2
        # dist21 = np.dot(l2, x1)
        # dist21 /= np.sqrt(dist21[:,0]**2+dist21[:,1]**2).reshape(-1,1)
        # matched_ids2 = np.argmin(dist21, axis=0)
        # total_dist12 = np.mean(dist21[matched_ids2, np.arange(x1.shape[1])])
        # serr = ep.Sampson_error(x1,x2,F)
        # vis.plot_epipolar_line(self.cameras[t1].img, self.cameras[t2].img, F, util.homogeneous(self.cameras[t1].get_gt_pts()), util.homogeneous(self.cameras[t2].get_gt_pts()))
        pts1, pts2 = self.cameras[t1].get_gt_pts(), self.cameras[t2].get_gt_pts()
        if self.settings['undist_points']:
            # undistort the ground truth matches
            pts1 = self.cameras[t1].undist_point(pts1, self.settings['undist_method']).T
            pts2 = self.cameras[t2].undist_point(pts2, self.settings['undist_method']).T

        vis.plotEpiline(self.cameras[t1].img, self.cameras[t2].img, pts1.astype(int), pts2.astype(int), F)
        # vis.plot_epipolar_line(self.cameras[t1].img, self.cameras[t2].img, F, util.homogeneous(self.detections[t1][1:]), util.homogeneous(self.detections[t2][1:]))

        # synchronization
        # Truncate detections
        self.cut_detection(second=self.settings['cut_detection_second'])
        # Add prior alpha
        self.init_alpha()
        # synchronize between detections
        # self.time_shift_from_F(F)
        # self.time_shift_from_pose()
        self.time_shift()
        # convert detection timestamps to global
        self.detection_to_global()

        # t1, t2 = self.sequence[0], self.sequence[1]

        # Find correspondences
        if self.cameras[t1].fps > self.cameras[t2].fps:
            d1, d2 = util.match_overlap(self.detections_global[t1], self.detections_global[t2])
        else:
            d2, d1 = util.match_overlap(self.detections_global[t2], self.detections_global[t1])

        # draw matches between dections
        if self.settings['undist_points']:
            # the background images are the original ones and are not undistorted, the detections need to be distorted
            d1_dist = self.cameras[t1].dist_point2d(d1[1:], method=self.settings['undist_method'])
            d2_dist = self.cameras[t2].dist_point2d(d2[1:], method=self.settings['undist_method'])

            vis.draw_detection_matches(self.cameras[t1].img, np.vstack([d1[0], d1_dist]), self.cameras[t2].img, np.vstack([d2[0],d2_dist]))
        else:
            vis.draw_detection_matches(self.cameras[t1].img, d1, self.cameras[t2].img, d2)

        # X, P, inlier, mask = ep.epipolar_pipeline_from_E(d1[1:], d2[1:], K1, K2, E)
        X, P, inlier, mask = ep.epipolar_pipeline_from_F(d1[1:], d2[1:], K1, K2, F)
        
        self.traj = np.vstack((d1[0][inlier==1][mask],X[:-1]))

    def init_static(self, error=10, inlier_only=False, debug=False):
        
        self.select_next_camera_static(init=True, debug=debug)
        
        t1, t2 = self.sequence[0], self.sequence[1]
        K1, K2 = self.cameras[t1].K, self.cameras[t2].K

        if debug:
            # in debug, use static ground truth as 2d featues
            if self.settings['undist_points']:
                # undistort the ground truth matches
                pts1 = self.cameras[t1].undist_point(self.cameras[t1].gt_pts.T, self.settings['undist_method']).T
                pts2 = self.cameras[t2].undist_point(self.cameras[t2].gt_pts.T, self.settings['undist_method']).T

                # plot the ground truth matches
                # FIXME: check pts_dist == gt_pts
                pts1_dist = self.cameras[t1].dist_point2d(pts1.T, method=self.settings['undist_method'])
                pts2_dist = self.cameras[t2].dist_point2d(pts2.T, method=self.settings['undist_method'])
                # vis.draw_detection_matches(self.cameras[t1].img, np.vstack([np.zeros(pts1_dist.shape[1]),pts1_dist]), self.cameras[t2].img, np.vstack([np.zeros(pts2_dist.shape[1]),pts2_dist]))
                vis.draw_detection_matches(self.cameras[t1].img, np.vstack([np.zeros(pts1.shape[0]),self.cameras[t1].gt_pts.T]), self.cameras[t2].img, np.vstack([np.zeros(pts2.shape[0]),self.cameras[t2].gt_pts.T]))

            else:
                pts1 = self.cameras[t1].gt_pts
                pts2 = self.cameras[t2].gt_pts
            
                vis.draw_detection_matches(self.cameras[t1].img, np.vstack([np.zeros(pts1.shape[0]),pts1.T]), self.cameras[t2].img, np.vstack([np.zeros(pts2.shape[0]),pts2.T]))

            # save the matching result, the indices of the ground truth matches are shared across all cameras
            query_ids = np.arange(self.cameras[t1].gt_pts.shape[0])
            train_ids = np.arange(self.cameras[t2].gt_pts.shape[0])

            # initialize the matrix for storing matching result
            match_res1 = -np.ones((self.numCam, self.cameras[t1].gt_pts.shape[0]))
            match_res2 = -np.ones((self.numCam, self.cameras[t2].gt_pts.shape[0]))
        
        else:
            # Match features
            pts1, pts2, matches, matchesMask = ep.matching_feature(self.cameras[t1].kp, self.cameras[t2].kp, self.cameras[t1].des, self.cameras[t2].des, ratio=0.8)
            # draw matches
            vis.draw_matches(self.cameras[t1].img, self.cameras[t1].kp, self.cameras[t2].img, self.cameras[t2].kp, matches, matchesMask)
            # get the valid matches
            match_ids = np.where(np.array(matchesMask)[:, 0] == 1)[0]

            # undistort the matched keypoints
            if self.settings['undist_points']:
                pts1 = self.cameras[t1].undist_point(np.array(pts1).T, self.settings['undist_method']).T
                pts2 = self.cameras[t2].undist_point(np.array(pts2).T, self.settings['undist_method']).T

            # get the valid matches
            query_ids = np.array([m[0].queryIdx for m in matches])
            train_ids = np.array([m[0].trainIdx for m in matches])
            query_ids = query_ids[match_ids]
            train_ids = train_ids[match_ids]

            # initialize the matrix for storing matching result
            match_res1 = -np.ones((self.numCam, len(self.cameras[t1].kp)))
            match_res2 = -np.ones((self.numCam, len(self.cameras[t2].kp)))

        # save the matching result to feature_dict
        match_res1[t2,query_ids] = train_ids
        match_res2[t1,train_ids] = query_ids
        self.feature_dict[t1] = match_res1
        self.feature_dict[t2] = match_res2
        
        pts1 = np.int32(pts1).T
        pts2 = np.int32(pts2).T

        # go through the epipolar pipeline for pose estimation and initial scene reconstruction
        # if debug:
        #     X, P, inlier, mask = ep.epipolar_pipeline(pts1, pts2, K1, K2, error, False, self.cameras[t1].img, self.cameras[t2].img)
        # else:
        #     X, P, inlier, mask = ep.epipolar_pipeline(pts1, pts2, K1, K2, error, inlier_only, self.cameras[t1].img, self.cameras[t2].img)
        
        X, P, inlier, mask = ep.epipolar_pipeline(pts1, pts2, K1, K2, error, inlier_only, self.cameras[t1].img, self.cameras[t2].img)

        # save the static part
        self.static = X[:-1]
        self.inlier_mask = np.ones(self.static.shape[1])

        # ids of the inliers
        inlier_ids = np.where(inlier == 1)[0]
        inlier_ids_masked = inlier_ids[mask]

        # also draw the inlier matches
        if not debug:
            matchesMask_inliers = np.zeros((len(matches), 2))
            match_ids_inliers = match_ids[inlier_ids_masked]
            matchesMask_inliers[match_ids_inliers] = [1, 0]

            vis.draw_matches(self.cameras[t1].img, self.cameras[t1].kp, self.cameras[t2].img, self.cameras[t2].kp, matches, matchesMask_inliers)

        # register these points and store their indices to the cameras
        self.cameras[t1].index_registered_2d = query_ids[inlier_ids_masked]
        self.cameras[t1].index_2d_3d = np.arange(self.static.shape[1])
        self.cameras[t2].index_registered_2d = train_ids[inlier_ids_masked]
        self.cameras[t2].index_2d_3d = np.arange(self.static.shape[1])

        # construct projections
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
    
    def error_cam_static(self, cam_id, mode='dist', norm=False, debug=False):
        '''
        Compute the reprojection error for static scene
        '''
        # get the 3D static points reconstructed from this camera
        point_3D = np.empty([3, 0])
        point_3D = np.hstack([point_3D, self.static[:, self.cameras[cam_id].index_2d_3d]])
        X = util.homogeneous(point_3D)
        
        # get the corresponding 2d static points
        if debug:
            # use the ground truth static poitns
            x = self.cameras[cam_id].get_gt_pts()
        else:
            # use the extracted static features
            x = self.cameras[cam_id].get_points()
        
        if self.settings['undist_points']:
            # undistort 2d points
            x = self.cameras[cam_id].undist_point(x, self.settings['undist_method'])
        
        x_cal = self.cameras[cam_id].projectPoint(X)

        # # distort point
        # x_cal = self.cameras[cam_id].dist_point3d(point_3D, self.settings['undist_method'])
        # print(cam_id,self.cameras[cam_id].index_2d_3d)
        
        #Normalize Tracks
        if norm:
            x_cal = np.dot(np.linalg.inv(self.cameras[cam_id].K), x_cal)
            x = np.dot(np.linalg.inv(self.cameras[cam_id].K), util.homogeneous(x))

        if mode == 'dist':
            return ep.reprojection_error(x, x_cal)
        elif mode == 'xy_1D':
            return np.concatenate((abs(x_cal[0] - x[0]), abs(x_cal[1] - x[1])))
        elif mode == 'xy_2D':
            return np.vstack((abs(x_cal[0] - x[0]), abs(x_cal[1] - x[1])))
        elif mode == 'each':
            error_x = np.zeros_like(x[0])
            error_y = np.zeros_like(x[0])
            error_x = abs(x_cal[0] - x[0])
            error_y = abs(x_cal[1] - x[1])
            return np.concatenate((error_x, error_y))
    

    def error_motion(self,cams,mode='dist',norm=False,motion_weights=0,motion_reg = False,motion_prior = False):
        '''
        Calculate the reprojection errors for a given camera for a multi_spline object. 

        - Accounts for motion prior

        - Motion priors available: 'F', 'KE'

        - computes error for motion prior regularization terms 
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
                    assert self.settings['motion_type'] == 'F' or self.settings['motion_type'] == 'KE','Motion type must be either F or KE' 
                    if self.settings['motion_type'] == 'F':
                        traj_ts = np.concatenate((traj_ts, traj_part[0,1:-1]))  
                    elif self.settings['motion_type'] == 'KE':
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


    def BA(self, numCam, max_iter=10, rs=False, motion_prior=False,motion_reg=False,motion_weights=1,norm=False,rs_bounds=False, debug=False):
        '''
        Bundle Adjustment with multiple splines

        The camera order is assumed to be the same as self.sequence
        '''

        def error_BA(x):
            '''
            Input is the model (parameters that need to be optimized)
            '''

            # if the static part are included, they are added to the beginning of the columns.
            # first parse the pararmeters for the static points
            if 'include_static' in self.settings.keys() and self.settings['include_static']:
                # the rest of the parameters are the 3d positions of the static scene
                self.static[:, self.inlier_mask > 0] = x[:num_3d_points * 3].reshape(-1, 3).T

            # Assign parameters to the class attributes
            sections = [numCam, numCam*2, numCam*3, numCam*3+numCam*num_camParam]
            # parse the rest of the parameters terms excluding the static parts
            model_parts = np.split(x[num_3d_points * 3:], sections)
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
            
            # also add the errors regarding the static part
            if 'include_static' in self.settings.keys() and self.settings['include_static']:
                for i in range(numCam):
                    error_each_static = self.error_cam_static(self.sequence[i], mode='each',debug=debug)
                    error = np.concatenate((error, error_each_static))
            
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
                try:
                    jac_cam[:,[i,i+numCam]] = 1 if self.settings['opt_sync'] else 0
                except:
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
            
            # add jacobian matrix for the static error terms
            if 'include_static' in self.settings.keys() and self.settings['include_static']:
                jac_parts = []
                # inlier ids of the static points used for BA
                inlier_ids = np.where(self.inlier_mask > 0)[0]
                # loop through all cameras
                for i in range(numCam):
                    # get camera id
                    cam_id = self.sequence[i]
                    # get the number of registered 2d points in this camera
                    num_pts_2d = self.cameras[cam_id].index_registered_2d.shape[0]
                    # the index of the registered 2d in 3d static points in the inlier set
                    registered_ids = np.in1d(inlier_ids, self.cameras[cam_id].index_2d_3d).nonzero()[0]

                    # initialize the jac mat (the structures of the sparsity matrix are the same in x and y direction)
                    jac_part = lil_matrix((num_pts_2d, num_param))

                    # mark all entries relate to this camera as 1 (the first num_3d_points*3 columns are for the static 3d points)
                    start, end = num_3d_points * 3 + i * num_camParam, num_3d_points * 3 + (i+1) * num_camParam
                    jac_part[:, start:end] = 1

                    # mark all entries relate to the static 3d points as 1
                    jac_part[np.arange(num_pts_2d), registered_ids * 3] = 1
                    jac_part[np.arange(num_pts_2d), registered_ids * 3 + 1] = 1
                    jac_part[np.arange(num_pts_2d), registered_ids * 3 + 2] = 1

                    # append mat for x direction and y direction
                    jac_parts.append(jac_part)
                    jac_parts.append(jac_part)

                    # FIXME: check if the dimension is the same
                
                jac_static = vstack(jac_parts)
                jac = vstack((jac, jac_static))
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
        
        # add the static part to BA
        num_3d_points = 0
        if 'include_static' in self.settings.keys() and self.settings['include_static']:
            num_3d_points = self.static[:, self.inlier_mask > 0].shape[1]
            model_static = self.static[:, self.inlier_mask > 0].T.ravel()
            # concatenate the static point_3d to the beginning
            model = np.concatenate((model_static, model))
        
        print('Number of BA parameters is {}'.format(len(model)))

        # constrain rs params to between 0 and 1
        if rs_bounds:
            l_bounds = np.ones((model.shape[0])) * -np.inf
            u_bounds = np.ones((model.shape[0])) * np.inf
            l_bounds[2*numCam:numCam*3] = 0
            u_bounds[2*numCam:numCam*3] = 1
            bounds_rs = (l_bounds, u_bounds)
        else:
            bounds_rs = (-np.inf,np.inf)

        # Set the Jacobian matrix
        A = jac_BA()

        '''Compute BA'''
        print('Doing BA with {} cameras...\n'.format(numCam))
        fn = lambda x: error_BA(x)
        # ignore jac_sparsity matrix for now for the static part
        if 'include_static' in self.settings.keys() and self.settings['include_static']:
            res = least_squares(fn,model,tr_solver='lsmr',xtol=1e-12,max_nfev=max_iter,verbose=1,bounds=bounds_rs)
        else:
            res = least_squares(fn,model,jac_sparsity=A,tr_solver='lsmr',xtol=1e-12,max_nfev=max_iter,verbose=1,bounds=bounds_rs)

        '''After BA'''
        # if the static part are included, they are added to the beginning of the columns.
        # first parse the pararmeters for the static points
        if 'include_static' in self.settings.keys() and self.settings['include_static']:
            # update the 3d positions of the static scene
            self.static[:, self.inlier_mask > 0] = res.x[:num_3d_points * 3].reshape(-1, 3).T

        # Assign the optimized model to alpha, beta, cam, and spline
        sections = [numCam, numCam*2, numCam*3, numCam*3+numCam*num_camParam]
        # exclude the parameters regarding the static part and parse the results
        model_parts = np.split(res.x[num_3d_points * 3:], sections)
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
    
    def BA_static(self, numCam, max_iter=10, debug=False):
        '''
        Function:
            standard BA with static points and camera models
        '''

        def error_BA(x):
            '''
            Function:
                define the error terms of BA
            Input:
                x = parameters to be optimized
            '''
            model_static, model_cam = x[:num_3d_points * 3], x[num_3d_points * 3:]

            # parse the 3d static
            self.static[:,self.inlier_mask > 0] = model_static.reshape(-1,3).T

            # parse the new camera parameters and poses
            cams = np.split(model_cam, numCam)
            for i in range(numCam):
                self.cameras[self.sequence[i]].vector2P(cams[i], calib=self.settings['opt_calib'])
            
            # compute the error terms
            error = np.array([])
            for i in range(numCam):
                error_static_each = self.error_cam_static(self.sequence[i], mode='each', debug=debug)
                error = np.concatenate((error, error_static_each))
            
            return error
        
        def jac_BA():
            '''
            Function:
                define the sparse jaccobian matrix
            '''

            num_param = len(model)

            jac_parts = []
            # inlier ids of the static points used for BA
            inlier_ids = np.where(self.inlier_mask > 0)[0]
            # loop through all cameras
            for i in range(numCam):
                # get camera id
                cam_id = self.sequence[i]
                # get the number of registered 2d points in this camera
                num_pts_2d = self.cameras[cam_id].index_registered_2d.shape[0]
                # the index of the registered 2d in 3d static points in the inlier set
                registered_ids = np.in1d(inlier_ids, self.cameras[cam_id].index_2d_3d).nonzero()[0]

                # initialize the jac mat (the structures of the sparsity matrix are the same in x and y direction)
                jac_part = lil_matrix((num_pts_2d, num_param))

                # mark all entries relate to this camera as 1 (the first num_3d_points*3 columns are for the static 3d points)
                start, end = num_3d_points * 3 + i * num_camParam, num_3d_points * 3 + (i+1) * num_camParam
                jac_part[:, start:end] = 1

                # mark all entries relate to the static 3d points as 1
                jac_part[np.arange(num_pts_2d), registered_ids * 3] = 1
                jac_part[np.arange(num_pts_2d), registered_ids * 3 + 1] = 1
                jac_part[np.arange(num_pts_2d), registered_ids * 3 + 2] = 1

                # append mat for x direction and y direction
                jac_parts.append(jac_part)
                jac_parts.append(jac_part)

                # FIXME: check if the dimension is the same
            
            jac = vstack(jac_parts)
            return jac.toarray()

        ''' BEFORE BA '''
        # camera model
        model_cam = np.array([])
        num_camParam = 15 if self.settings['opt_calib'] else 6
        for i in self.sequence[:numCam]:
            model_cam = np.concatenate((model_cam, self.cameras[i].P2vector(calib=self.settings['opt_calib'])))
        
        # 3d static points
        num_3d_points = self.static[:, self.inlier_mask > 0].shape[1]
        model_static = self.static[:, self.inlier_mask > 0].T.ravel()
        # concatenate the static point_3d to the beginning
        model = np.concatenate((model_static, model_cam))

        print('Number of BA parameters is {}'.format(len(model)))

        ''' BA '''
        # set the jacobian matrix
        A = jac_BA()
        print('Doing BA with {} cameras and {} static points...\n'.format(numCam, num_3d_points))

        # define the error function
        fn = lambda x: error_BA(x)

        # least-sqaure optimization for BA
        # res = least_squares(fn, model, tr_solver='lsmr', xtol=1e-12, max_nfev=max_iter, verbose=1)
        res = least_squares(fn, model, jac_sparsity=A, tr_solver='lsmr', xtol=1e-12, max_nfev=max_iter, verbose=1)

        ''' AFTER BA '''
        # parse the result of BA
        res_static, res_cam = res.x[:num_3d_points * 3], res.x[num_3d_points * 3:]

        # update the 3d static points
        self.static[:, self.inlier_mask > 0] = res_static.reshape(-1,3).T

        # update the camera parameters
        cams = np.split(res_cam, numCam)
        for i in range(numCam):
            self.cameras[self.sequence[i]].vector2P(cams[i], calib=self.settings['opt_calib'])
            print(self.cameras[self.sequence[i]].K)
    
    def remove_outliers(self, cams, thres=30, verbose=False, debug=False):
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
                
                # if the static part is included, also remove outliers in the static part
                if 'include_static' in self.settings.keys() and self.settings['include_static']:
                    # filter out the outliers from the static scene
                    error_static = self.error_cam_static(i, mode='dist', debug=debug)

                    # indices of the outliers in the reconstructed 3D points
                    outlier_ids = self.cameras[i].index_2d_3d[error_static >= self.settings['thres_outlier_static']]
                    # maskout these points in the inlier_mask
                    self.inlier_mask[outlier_ids] == 0
                    # remove feature ids corresponds to the outliers from the registered list
                    self.cameras[i].index_2d_3d = self.cameras[i].index_2d_3d[error_static < self.settings['thres_outlier_static']]
                    self.cameras[i].index_registered_2d = self.cameras[i].index_registered_2d[error_static < self.settings['thres_outlier_static']]

                    # also update the registered 2d index of the other cameras
                    for j in cams:
                        if j == i:
                            continue

                        # find the ids of the outlier in the camera
                        cond = np.in1d(self.cameras[j].index_2d_3d, outlier_ids)
                        self.cameras[j].index_2d_3d = self.cameras[j].index_2d_3d[~cond]
                        self.cameras[j].index_registered_2d = self.cameras[j].index_registered_2d[~cond]
                    
                    if verbose:
                        print('{} out of {} static points are removed for camera {}'.format(len(outlier_ids), len(error_static), i))

    def remove_outliers_static(self, cams, thres=30, verbose=False, debug=False):
        '''
        Function: 
            remove the outliers from the static scene
        '''
        if thres:
            for i in cams:
                # filter out the outliers from the static scene
                error_static = self.error_cam_static(i, mode='dist', debug=debug)

                # indices of the outliers in the reconstructed 3D points
                outlier_ids = self.cameras[i].index_2d_3d[error_static >= self.settings['thres_outlier_static']]
                # maskout these points in the inlier_mask
                self.inlier_mask[outlier_ids] == 0
                # remove feature ids corresponds to the outliers from the registered list
                self.cameras[i].index_2d_3d = self.cameras[i].index_2d_3d[error_static < self.settings['thres_outlier_static']]
                self.cameras[i].index_registered_2d = self.cameras[i].index_registered_2d[error_static < self.settings['thres_outlier_static']]

                # also update the registered 2d index of the other cameras
                for j in cams:
                    if j == i:
                        continue

                    # find the ids of the outlier in the camera
                    cond = np.in1d(self.cameras[j].index_2d_3d, outlier_ids)
                    self.cameras[j].index_2d_3d = self.cameras[j].index_2d_3d[~cond]
                    self.cameras[j].index_registered_2d = self.cameras[j].index_registered_2d[~cond]
                
                if verbose:
                    print('{} out of {} static points are removed for camera {}'.format(len(outlier_ids), len(error_static), i))
    
    def register_new_camera_static(self, cam_id, cams, debug=False):
        '''
        Function:
            find 2d - 3d correspondences between the features in the new camera and existing 3d
        Input:
            cam_id = the id of the new camera
            cams = a sequence of existing cameras
        Output:
            pts_2d = the features in the new camera that match the existing 3d points (does not need to undistort the pts, this will be taken care when solving pnp)
            pts_3d = the matched existing 3d points
        '''
        if debug:
            # use the ground truth matches
            # if have not yet initialize the cam_id in the feature dict, do so
            if cam_id not in self.feature_dict.keys():
                # initialize the matching result of this new camera to be stored to the dict
                self.feature_dict[cam_id] = -np.ones((self.numCam, len(self.cameras[cam_id].gt_pts)))
                # match between the features of this new camera and all other cameras
                for i in cams:
                    if i == cam_id:
                        continue
                    # all the ground truth matches are mutually valid, so directly update
                    self.feature_dict[cam_id][i] = np.arange(len(self.cameras[i].gt_pts))
                    self.feature_dict[i][cam_id] = np.arange(len(self.cameras[cam_id].gt_pts))
            
            # get the registered indices based on matches in feature dict
            for i in cams:
                if i == cam_id:
                    continue
                # update the registered indices (all ground truth matches share the same indices)
                self.cameras[cam_id].index_2d_3d = self.cameras[i].index_2d_3d
                self.cameras[cam_id].index_registered_2d = self.cameras[i].index_registered_2d
            # add the ground truth
            pts_2d = self.cameras[cam_id].get_gt_points()

        else:
            # get the features and descriptors of the new camera
            kp1, des1 = self.cameras[cam_id].kp, self.cameras[cam_id].des

            # if have not yet initialize cam_id in feature_dict, do so
            if cam_id not in self.feature_dict.keys():
                # initilize the matching result of the new camera to be stored in feature_dict
                match_res = -np.ones((self.numCam, len(kp1)))
                # loop through all the cameras
                for i in cams:
                    if i == cam_id:
                        continue
                    # get the features of the old camera
                    kp2, des2 = self.cameras[i].kp, self.cameras[i].des
                    # match the features of the new camera and the old camera
                    _, _, matches, matchesMask = ep.matching_feature(kp1, kp2, des1, des2, ratio=0.8)
                    # draw matches
                    vis.draw_matches(self.cameras[cam_id].img, self.cameras[cam_id].kp, self.cameras[i].img, self.cameras[i].kp, matches, matchesMask)
                    
                    # get the matched indices
                    query_ids = np.array([m[0].queryIdx for m in matches])
                    train_ids = np.array([m[0].trainIdx for m in matches])

                    # get the valid matches
                    match_ids = np.where(np.array(matchesMask)[:, 0] == 1)[0]
                    query_ids = query_ids[match_ids]
                    train_ids = train_ids[match_ids]

                    # save the matching result to feature_dict
                    match_res[i, query_ids] = train_ids
                    self.feature_dict[cam_id] = match_res
                    self.feature_dict[i][cam_id, train_ids] = query_ids

            # otherwise get the matching results from feature dict
            for i in cams:
                if i == cam_id:
                    continue
                
                # get the matching results between cam_id and i
                match_res = self.feature_dict[cam_id][i]
                train_ids = match_res[match_res >= 0]
                query_ids = np.where(match_res >= 0)[0]
                # find the indices of the old features that has been used
                _, train_in_ids, registered_ids = np.intersect1d(train_ids, self.cameras[i].index_registered_2d, return_indices=True)
                
                # # register corresponding query_ids of the new camera
                # self.cameras[cam_id].index_registered_2d = np.union1d(self.cameras[cam_id].index_registered_2d, query_ids[train_in_ids]).astype(int)
                # self.cameras[cam_id].index_2d_3d = np.union1d(self.cameras[cam_id].index_2d_3d, self.cameras[i].index_2d_3d[registered_ids]).astype(int)
                
                # find the indices of the point that has not been added
                newids = np.isin(self.cameras[i].index_2d_3d[registered_ids], self.cameras[cam_id].index_2d_3d, invert=True)
                self.cameras[cam_id].index_2d_3d = np.concatenate([self.cameras[cam_id].index_2d_3d, self.cameras[i].index_2d_3d[registered_ids[newids]]]).astype(int)
                self.cameras[cam_id].index_registered_2d = np.concatenate([self.cameras[cam_id].index_registered_2d, query_ids[train_in_ids[newids]]]).astype(int)
            
            # get the registered 2d features from the new camera
            pts_2d = self.cameras[cam_id].get_points()
        
        # get the registered 3d static point from the new camera
        pts_3d = self.static[:, self.cameras[cam_id].index_2d_3d]

        return pts_2d, pts_3d
    
    def get_camera_pose_static(self, cam_id, cams, error=8, verbose=0, debug=False):
        '''
        Function:
            solve the PnP to get the pose of the new camera
            the distortion model is taken care by PnP
        Input:
            cam_id = the index of the new camera
            pts_2d = the matched raw 2d feature positions
            pts_3d = the matched reconstructed 3d points
        '''
        pts_2d, pts_3d = self.register_new_camera_static(cam_id, cams, debug)

        # PnP solution from OpenCV
        N = pts_3d.shape[1]
        objectPoints = np.ascontiguousarray(pts_3d.T).reshape((N,1,3))
        imagePoints  = np.ascontiguousarray(pts_2d.T).reshape((N,1,2))
        distCoeffs = self.cameras[cam_id].d
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, self.cameras[cam_id].K, distCoeffs, reprojectionError=error)

        # update the indices of the registered 2d features, removing the outliers
        self.cameras[cam_id].index_registered_2d = self.cameras[cam_id].index_registered_2d[inliers.ravel()]
        self.cameras[cam_id].index_2d_3d = self.cameras[cam_id].index_2d_3d[inliers.ravel()]

        self.cameras[cam_id].R = cv2.Rodrigues(rvec)[0]
        self.cameras[cam_id].t = tvec.reshape(-1,)
        self.cameras[cam_id].compose()

        if verbose:
            print('{} out of {} points are inliers for PnP'.format(inliers.shape[0], N))

    def get_camera_pose(self, cam_id, cams, error=8, verbose=0, debug=False):
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
        # FIXME: SHOULD USE THE RAW DETECTION? INSTEAD OF THE UNDISTORTED ONES??
        
        num_detect = detect.shape[1]
        # if the static part is also included, add the static part to the points as well
        if 'include_static' in self.settings.keys() and self.settings['include_static']:
            pts_2d, pts_3d = self.register_new_camera_static(cam_id, cams, debug)

            # stack the static points with the detections
            detect = np.hstack([detect, pts_2d])
            point_3D = np.hstack([point_3D, pts_3d])

        # PnP solution from OpenCV
        N = point_3D.shape[1]
        objectPoints = np.ascontiguousarray(point_3D.T).reshape((N,1,3))
        imagePoints  = np.ascontiguousarray(detect[1:].T).reshape((N,1,2))
        distCoeffs = self.cameras[cam_id].d
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, self.cameras[cam_id].K, distCoeffs, reprojectionError=error)

        # update the index of the registered 2d static points to remove the outliers
        self.cameras[cam_id].index_registered_2d = self.cameras[cam_id].index_registered_2d[inliers[inliers >= num_detect] - num_detect]
        self.cameras[cam_id].index_2d_3d = self.cameras[cam_id].index_2d_3d[inliers[inliers >= num_detect] - num_detect]

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
    
    def triangulate_static(self, cam_id, cams, thres=0, verbose=0):
        '''
        Triangulate new points from the static scene to the existing 3D scene

        cam_id is the new camera
        
        cams must be an iterable that contains cameras that have been processed to build the 3D static scene
        '''

        assert self.cameras[cam_id].P is not None, 'The camera pose must be computed first'

        # find the features of this camera that have not yet been triangulated
        all_ids = np.arange(len(self.cameras[cam_id].kp))
        cand_ids = all_ids[~np.in1d(all_ids, self.cameras[cam_id].index_registered_2d)]

        X_new = np.empty([3, 0])
        added_cand_ids = np.empty(0)
        # loop through all old cameras, and triangulate the matched features
        for i in cams:
            if i == cam_id:
                continue
            # get the matched features in this camera
            new_ids, matched_ids, _ = np.intersect1d(self.feature_dict[i][cam_id], cand_ids, return_indices=True)
            new_ids = new_ids.astype(int)
            # get the 2d points from the two cameras
            pts1 = self.cameras[cam_id].get_points(new_ids)
            pts2 = self.cameras[i].get_points(matched_ids)
            
            # undistort the 2d points if needed
            if self.settings['undist_points']:
                pts1 = self.cameras[cam_id].undist_point(pts1, self.settings['undist_method'])
                pts2 = self.cameras[i].undist_point(pts2, self.settings['undist_method'])
            
            # get the poses of the two cameras
            P1, P2 = self.cameras[cam_id].P, self.cameras[i].P

            # triangulate the points
            X_i = ep.triangulate_matlab(pts1, pts2, P1, P2)

            # Check reprojection error directly after triangulation, preserve those with small error
            if thres:
                err_1 = ep.reprojection_error(pts1, self.cameras[cam_id].projectPoint(X_i))
                err_2 = ep.reprojection_error(pts2, self.cameras[i].projectPoint(X_i))
                mask = np.logical_and(err_1 < thres, err_2 < thres)
                X_i = X_i[:, mask]
                new_ids = new_ids[mask]
                matched_ids = matched_ids[mask]

                if verbose:
                    print('{} out of {} points are triangulated'.format(sum(mask), len(err_1)))
            
            added_mask = np.in1d(new_ids, added_cand_ids)
            # assign ids to the points to be added
            ids_3d_new = np.arange(new_ids[~added_mask].shape[0]) + X_new.shape[1] + int(np.sum(self.inlier_mask))
            if verbose:
                print('{} new points are added'.format(ids_3d_new.shape[0]))
            
            # registered the points that have not yet been added
            self.cameras[cam_id].index_registered_2d = np.concatenate((self.cameras[cam_id].index_registered_2d, new_ids[~added_mask]))
            self.cameras[cam_id].index_2d_3d = np.concatenate((self.cameras[cam_id].index_2d_3d, ids_3d_new))

            # register the new points in the old camera
            # first add the points which have previously been added
            self.cameras[i].index_registered_2d = np.concatenate((self.cameras[i].index_registered_2d, matched_ids[added_mask]))
            # get the corresponding 3d point indices
            ids_3d_old = np.in1d(added_cand_ids, new_ids).nonzero()[0] + int(np.sum(self.inlier_mask))
            self.cameras[i].index_2d_3d = np.concatenate((self.cameras[i].index_2d_3d, ids_3d_old))
            # then add the points which have not yet been added
            self.cameras[i].index_registered_2d = np.concatenate((self.cameras[i].index_registered_2d, matched_ids[~added_mask]))
            self.cameras[i].index_2d_3d = np.concatenate((self.cameras[i].index_2d_3d, ids_3d_new))
            
            # keep a track of the new ids 
            add_cand_ids = np.concatenate((added_cand_ids, new_ids[~added_mask]))
            # add the new points to the record
            X_new = np.hstack([X_new, X_i[:-1, ~added_mask]])
        
        # add these new points to the static scene
        self.static = np.hstack([self.static, X_new])
        self.inlier_mask = np.concatenate((self.inlier_mask, np.ones(X_new.shape[1])))

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
    
    def select_next_camera_static(self, init=False, debug=False):
        '''
        Function:
            find to the next camera to be registered based on the static part only
        Input:
            init = whether in the initialization or not
        '''

        assert self.numCam >= 2

        if not self.find_order:
            return
        
        if init:
            # take the first two cameras as initial pairs
            self.sequence = [0,1]
        else:
            # get the candidate new cameras
            candidate = []
            for i in range(self.numCam):
                if self.cameras[i].P is None:
                    candidate.append(i)
            
            # find the next camera with the largest overlap
            max_overlap = 0
            for cam_id in candidate:
                num_overlap = 0
                overlap_ids = np.empty(0)
                # initialize feature_dict
                if debug:
                    # use the ground truth matches
                    # initialize the matching result of this new camera to be stored to the dict
                    self.feature_dict[cam_id] = -np.ones((self.numCam, len(self.cameras[cam_id].gt_pts)))
                    # match between the features of this new camera and all other cameras
                    for i in self.sequence:
                        # all ground truth matches are mutually valid, directly populate with indices
                        self.feature_dict[cam_id][i] = np.arange(len(self.cameras[i].gt_pts))
                        self.feature_dict[i][cam_id] = np.arange(len(self.cameras[cam_id].gt_pts))

                        # get the registered indices in 3d
                        overlap_ids = np.union1d(overlap_ids,self.cameras[i].index_2d_3d)
                        # sum the number of registered indices
                        num_overlap = len(overlap_ids)
                else:
                    # get the features and descriptors of the new camera
                    kp1, des1 = self.cameras[cam_id].kp, self.cameras[cam_id].des

                    # initilize the matching result of the new camera to be stored in feature_dict
                    match_res = -np.ones((self.numCam, len(kp1)))
                    # loop through all the cameras
                    for i in self.sequence:
                        # get the features of the old camera
                        kp2, des2 = self.cameras[i].kp, self.cameras[i].des
                        # match the features of the new camera and the old camera
                        _, _, matches, matchesMask = ep.matching_feature(kp1, kp2, des1, des2, ratio=0.8)
                        # draw matches
                        vis.draw_matches(self.cameras[cam_id].img, self.cameras[cam_id].kp, self.cameras[i].img, self.cameras[i].kp, matches, matchesMask)

                        # get the matched indices
                        query_ids = np.array([m[0].queryIdx for m in matches])
                        train_ids = np.array([m[0].trainIdx for m in matches])

                        # get the valid matches
                        match_ids = np.where(np.array(matchesMask)[:, 0] == 1)[0]
                        query_ids = query_ids[match_ids]
                        train_ids = train_ids[match_ids]

                        # save the matching result to feature_dict
                        match_res[i, query_ids] = train_ids
                        self.feature_dict[cam_id] = match_res
                        self.feature_dict[i][cam_id, train_ids] = query_ids

                        # find the indices of the old features that has been used
                        _, _, registered_ids = np.intersect1d(train_ids, self.cameras[i].index_registered_2d, return_indices=True)
                        # find the corresponding indices in 3D points
                        overlap_ids = np.union1d(overlap_ids, self.cameras[i].index_2d_3d[registered_ids])
                        num_overlap = len(overlap_ids)
                    
                if num_overlap > max_overlap:
                    next_cam = cam_id

            # add the next cam to the sequence
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
# FIXME: threshold=20
                print('Status: {} from {} cam finished'.format(j+1,self.numCam))
            self.beta = beta
            self.beta_after_Fbeta = beta.copy()
    
    def time_shift_from_F(self, F):
        '''
        This function computes relative time shifts of each camera to the ref camera using the given corresponding frame numbers

        If the given frame indices are precise, then the time shifts are directly transformed from them.
        '''

        assert len(self.cf)==self.numCam, 'The number of frame indices should equal to the number of cameras'

        if self.settings['cf_exact']:
            self.beta = self.cf[self.ref_cam] - self.alpha*self.cf
            print('The given corresponding frames are directly exploited as temporal synchronization\n')
        else:

            print('Computing temporal synchronization...\n')
            beta = np.zeros(self.numCam)
            i = self.ref_cam
            for j in range(self.numCam):
                if j==i:
                    beta[j] = 0
                else:
                    beta[j], _ = sync.sync_bf_from_F(self.cameras[i].fps, self.cameras[j].fps,
                                                    self.detections[i], self.detections[j],
                                                    self.cf[i], self.cf[j], F)
                print('Status: {} from {} cam finished'.format(j+1,self.numCam))
            self.beta = beta
            self.beta_after_Fbeta = beta.copy()
    
    def time_shift_from_pose(self):
        '''
        This function computes relative time shifts of each camera to the ref camera using the given corresponding frame numbers

        If the given frame indices are precise, then the time shifts are directly transformed from them.
        '''

        assert len(self.cf)==self.numCam, 'The number of frame indices should equal to the number of cameras'

        print('Computing temporal synchronization...\n')
        beta = np.zeros(self.numCam)
        i = self.ref_cam
        for j in range(self.numCam):
            if j==i:
                beta[j] = 0
            else:
                beta[j], _ = sync.sync_bf_from_pose(self.cameras[i].fps, self.cameras[j].fps,
                                                self.detections[i], self.detections[j],
                                                self.cf[i], self.cf[j], self.cameras[i], self.cameras[j])
            print('Status: {} from {} cam finished'.format(j+1,self.numCam))
        self.beta = beta
        self.beta_after_Fbeta = beta.copy()

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
        
        # information for the static part 
        self.img_path = kwargs.get('img_path')
        self.img = None
        self.kp = []
        self.des = []
        # the indices of the features used for 3D static point reconstruction
        self.index_registered_2d = np.empty(0)
        # the indices of the 3D static points that corresponds to the used feautures
        self.index_2d_3d = np.empty(0)
        # ground truth static matches for debugging
        self.gt_pts = None

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

    def undist_point(self, points, method='opencv'):
        '''
        Function:
            A wrapper function for the undistort point function with different models
        Input:
            points = points to be undistorted
            method = distortion model ['opencv', 'division']
        Output:
            a sets of undistorted points
        '''

        if method == 'division':
            return self.undist_point_div(points)
        else:
            return self.undist_point_opencv(points)
    

    def undist_point_opencv(self,points):
        
        assert points.shape[0]==2, 'Input must be a 2D array'

        num = points.shape[1]

        src = np.ascontiguousarray(points.T).reshape((num,1,2))
        dst = cv2.undistortPoints(src, self.K, self.d)
        dst_unnorm = np.dot(self.K, util.homogeneous(dst.reshape((num,2)).T))

        return dst_unnorm[:2]
    
    def undist_point_div(self, points):
        '''
        CURRENTLY NOT AVAILABLE YET
        '''
        return points[:2]

    def dist_point3d(self, points, method='opencv'):
        return self.dist_point3d_opencv(points)
    
    def dist_point3d_opencv(self, points):
        pts_dist,_ = cv2.projectPoints(points, self.R, self.t, self.K, self.d)
        return pts_dist.reshape(-1,2).T

    def dist_point2d(self, points, method='opencv'):
        return self.dist_point2d_opencv(points)

    def dist_point2d_opencv(self, points):
        pts_dist, _ = cv2.projectPoints(np.dot(np.linalg.inv(self.K), util.homogeneous(points)), np.eye(3), np.zeros((3,1)), self.K, self.d)
        return pts_dist.reshape(-1,2).T


    def info(self):
        print('\n P:')
        print(self.P)
        print('\n K:')
        print(self.K)
        print('\n R:')
        print(self.R)
        print('\n t:')
        print(self.t)
    
    def read_img(self):
        self.img = cv2.imread(self.img_path)
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def extract_features(self, img=None, method='sift'):
        '''
        Function:
            extract features from the images and store the featues to the class
        Input:
            img = image of which features are to be extracted
                  (if None given, then extract the features of this camera)
            method = feature extractor
        '''
        if img is None:
            self.read_img()
            self.kp, self.des = ep.extract_SIFT_feature(self.img)
        else:
            print("extract features from given img")
            self.kp, self.des = ep.extract_SIFT_feature(img)

    def get_points(self, indices=None):
        '''
        get the registered 2d points (2, n)
        '''
        point_2d = np.empty([2, 0])
        if indices is not None:
            return np.hstack([point_2d, np.array([self.kp[idx].pt for idx in indices]).T.reshape(2,-1)])
        return np.hstack([
            point_2d,
            np.array([self.kp[idx].pt
                    for idx in self.index_registered_2d]).T.reshape(2, -1)
        ])

    def get_gt_pts(self):
        '''
        get the registered 2d ground truth points (2, n)
        '''
        return self.gt_pts[self.index_registered_2d].T
    
    def unpack_sift_kp(self):
        self.kp = np.array([kp.pt for kp in self.kp])

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
    for i, path in enumerate(path_cam):
        try:
            with open(path, 'r') as file:
                cam = json.load(file)
        except:
            raise Exception('Wrong input of camera')

        if len(cam['distCoeff']) == 4:
            cam['distCoeff'].append(0)

        # load camera information
        camera = Camera(K=np.asfarray(cam['K-matrix']), d=np.asfarray(cam['distCoeff']), fps=cam['fps'], resolution=cam['resolution'], img_path=cam['img_path'])
        # extract features
        if 'include_static' in config['settings'].keys() and config['settings']['include_static']:
            camera.extract_features(method=flight.settings['feature_extractor'])
        else:
            camera.read_img()

        # load the ground truth static matches if given
        if 'optional inputs' in config.keys() and 'static_ground_truth' in config['optional inputs'].keys():
            camera.gt_pts = np.loadtxt(config['optional inputs']['static_ground_truth'][i])
        
        flight.addCamera(camera)

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
    if 'optional inputs' in config.keys():
        if 'ground_truth' in config['optional inputs'].keys():
            flight.gt = config['optional inputs']['ground_truth']
        if 'static_ground_truth_3d' in config['optional inputs'].keys():
            flight.gt_static = np.loadtxt(config['optional inputs']['static_ground_truth_3d'])

    print('Input data are loaded successfully, a scene is created.\n')
    return flight
