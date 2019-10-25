import numpy as np
import os
import pickle
import math
from datetime import datetime as dt, timedelta
#from read_data import readFile
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree as KDTree
from transformations import affine_matrix_from_points as affine_matrix_from_points 
from transformations import decompose_matrix as decompose_matrix
from cv2 import estimateAffine3D
from cv2 import ppf_match_3d_ICP
import random
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler
import collections
import visualization as vis
#from icp import icp
#from queue import PriorityQueue

class ROTNODE():
    def __init__(self):
        self.a = 0.
        self.b = 0.
        self.c = 0.
        self.w = 0.
        self.ub = 0.
        self.lb = 0.
        self.l = 0

    def __lt__(self, a):
        if self.lb != a.lb:
            return self.lb > a.lb
        else:
            return self.w < a.w
    
    def __str__(self):
        return 'a: {}, b: {}, c: {}, w: {}, l: {}, ub: {}, lb: {}'.format(self.a,
            self.b, self.c, self.w, self.l, self.ub, self.lb)


class TRANSNODE():
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.w = 0.
        self.ub = 0.
        self.lb = 0.

    def __lt__(self, a):
        if self.lb != a.lb:
            return self.lb > a.lb
        else:
            return self.w < a.w

    def __str__(self):
        return 'x: {}, y: {}, z: {}, w: {}, ub: {}, lb: {}'.format(self.x,
            self.y, self.z, self.w, self.ub, self.lb)

def standardize(p):
    scaler = StandardScaler().fit(p)
    return scaler.transform(p)

    #scaler.fit(dst)
    #src_scale = preprocessing.scale(src) # scaler.transform(src)

def distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1-p2),axis=-1))

def L2_error(p1, p2):
    return np.sum(np.square(distance(p1,p2)))

def lists_to_dict(coded, plain):
    '''
    (list of str, list of str) -> dict of {str: str}
    Return a dict in which the keys are the items in coded and
    the values are the items in the same positions in plain. The
    two parameters must have the same length.

    >>> d = lists_to_dict(['a', 'b', 'c', 'e', 'd'],  ['f', 'u', 'n', 'd', 'y'])
    >>> d == {'a': 'f', 'b': 'u', 'c': 'n', 'e': 'd', 'd': 'y'}
    True
    '''
    dic = {}
    dic = {key:value for key, value in zip(coded, plain)}

    #x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    inds = []
    sorted_list = sorted(dic.items(), key=lambda kv: kv[0])
    for key, value in sorted_list:
        inds.append(value)
    return sorted_list,inds

def Q(p, x):
    u_p1 = np.mean(p[:,:3],axis=0)#.reshape((3,1))
    u_p2 = np.mean(x[:,:3],axis=0)#.reshape((3,1))

    u_p1p2 = u_p1.dot(u_p2.T)
    points1 = p[:,:3]#.reshape((p.shape[0],3,1))
    points2 = x[:,:3]#.reshape((x.shape[0],3,1))
    p1 = points1 - u_p1
    p2 = points2 - u_p2

    p1p2 = p1.T.dot(p2)
    u,s,v = np.linalg.svd(p1p2)

    R = v.T.dot(u.T)

    # if np.linalg.det(R) < 0:
    # 	v[2,:] *= -1
    # 	R = v.T.dot(u.T)

    det = np.linalg.det(R)
    temp = np.eye(3)
    temp[2,2] = det

    R = v.T.dot(temp.dot(u.T))

    u_p1 = u_p1.reshape((3,1))
    u_p2 = u_p2.reshape((3,1))
    t = u_p2 - R.dot(u_p1)

    X = np.eye(4,dtype=p.dtype)

    X[:3,:3] = R[:,:]
    X[:3,3] = t[:,0]
    R = X[:,:]
    
    return R, R[:3,:3], R[:3,3].reshape((3,1))

def closest_point(src,dst):
    
    #src = standardize(src)
    #dst = standardize(dst)
    #M0 = amp(src[:,:3].T,dst[:,:3].T,shear=False)
    #src = np.dot(M0,src.T).T

    # dst_scaler = StandardScaler()  
    # dst_scaler.fit(dst)
    # dst = dst_scaler.transform(dst)

    # src_scaler = StandardScaler()  
    # src_scaler.fit(src)
    # src = src_scaler.transform(src)

    #scaler.fit(dst)
    #src = preprocessing.scale(src) # scaler.transform(src)

    # dst = preprocessing.scale(dst) #scaler.transform(dst)
    # src = preprocessing.scale(src) #
    
    src_tree = KDTree(src.T)
    dst_tree = KDTree(dst.T)

    src_dist, src_ind = src_tree.query(dst.T)
    dst_dist, dst_ind = dst_tree.query(src.T)

    src_ind = src_ind.reshape((src_ind.shape[0]))
    dst_ind = dst_ind.reshape((dst_ind.shape[0]))

    src_dist = src_dist.reshape((src_dist.shape[0]))
    dst_dist = dst_dist.reshape((dst_dist.shape[0]))

    src_dict,min_src_inds = lists_to_dict(src_dist,src_ind)
    dst_dict,min_dst_inds = lists_to_dict(dst_dist,dst_ind)



    #min_ind = np.array(range(len(ind)))
    #ind_sample =random.sample(np.array(range(len(src_ind))).tolist(), minPoints)
    #dst_sample =random.sample(np.array(range(len(dst_ind))).tolist(), minPoints)
    #ind_sample.sort() #, p=[0.5, 0.1, 0.1, 0.3])
    #dst_sample.sort()
    #ind = random.sample(ind.tolist(),minPoints)
    #dist = dist.reshape((dist.shape[0]))
    # print(_[:10])
    minPoints = min(len(min_src_inds),len(min_dst_inds))

    src_inds = min_src_inds[:minPoints]
    dst_inds = min_dst_inds[:minPoints]

    src_guess = src[:,min_src_inds[:minPoints]]
    dst_guess = dst[:,min_dst_inds[:minPoints]]

    sg = src[:,src_ind]
    dg = dst[:,dst_ind]
    
    #rt_val,M, inliers = estimateAffine3D(srcp2[ind,:3],p1[:,:3])
    #M = affine_matrix_from_points(src_guess[:,:3].T,dst_guess[:,:3].T,shear=False,scale=True)
    #scale, shear, angles, translate, perspective = decompose_matrix(M)
    #del ind
    return sg,dg,src_ind,dst_ind

def ICP(src, dst, R=np.eye(3), T=np.zeros((3,1))):
    iters = 1000
    minPoints = min(dst.shape[1],src.shape[1])
    #tree = KDTree(src)
    #minPoints = min(dst.shape[0],src.shape[0])
    src0 = src.copy()#[:minPoints,:]
    #p2 = p2[:minPoints,:]
    M0 = np.eye(4, dtype=np.float32)
    r = R[:,:]
    t = T[:,:]
    M0[:3,:3] = r[:,:]
    M0[:3,3] = t[:,0]

    #Scale
    #src_scale = src0**2
    #dst_scale *= dst
    #M0[:3, :3] *= math.sqrt(np.sum(dst**2) / np.sum(src0**2))

    #src0 = M0.dot(src0.T)
    #src0 = src0.T

    #vis.show_trajectory_3D(src0.T,dst.T)
    # Scale Data for KNN Computation 
    #dst_scale = preprocessing.scale(dst) #scaler.transform(dst)
    #src_scale = preprocessing.scale(src0) # scaler.transform(src)

    sg,dg,src_ind,dst_ind = closest_point(src0, dst)
    M1 = affine_matrix_from_points(src0[:3,src_ind],dst[:3,dst_ind],shear=False,scale=True)
    scale, shear, angles, translate, perspective = decompose_matrix(M1)
    src0 = M1.dot(src0)
    #vis.show_tf_trajectory_3D(src0,X_dst=dst,color=False,line=False,pause=False)
    #vis.show_tf_trajectory_3D(src,X_dst=dst,color=False,line=False,pause=False)
    #src_guess = src_guess.T
    m_err = L2_error(src0[:3,src_ind],dst[:3,dst_ind])

    error = np.sqrt((src[0]-dst[0])**2 + (src[1]-dst[1])**2 + (src[2]-dst[2])**2)
    #print('Mean error between transformed reconstruction and GPS data: {:.6f}, unit is meter.'.format(np.mean(error)))

    error_icp = np.sqrt((src0[0]-dst[0])**2 + (src0[1]-dst[1])**2 + (src0[2]-dst[2])**2)
    #print('Mean ICP error between transformed reconstruction and GPS data: {:.6f}, unit is meter.'.format(np.mean(error_icp)))

    error_icp = np.sqrt((sg[0]-dg[0])**2 + (sg[1]-dg[1])**2 + (sg[2]-dg[2])**2)
    #print('Mean selected ICP error between transformed reconstruction and GPS data: {:.6f}, unit is meter.'.format(np.mean(error_icp)))
    
    #src0 = sg
    del sg #src_guess
    del dg #dst_guess
    
    #print('Initial Error: {}'.format(m_err))
    for i in range(iters):
        sg,dg,src_ind,dst_ind = closest_point(src0,dst)
        M = affine_matrix_from_points(src0[:3,src_ind],dst[:3,dst_ind],shear=False,scale=True)
        #M = affine_matrix_from_points(sg[:3,src_ind],dg[:3,dst_ind],shear=False,scale=True)
        #rt_val,X0, inliers = estimateAffine3D(Y1[:,:3],p0[:,:3])
        #X0 = amp(Y[:,:3].T,p0[:,:3].T,shear=True,scale=True)
              
        #X0, R0, t0 = Q(p0, Y)
        #r = R0.dot(r)
        #t = R0.dot(t) + t0
        #src0 = np.dot(M,src0.T)
        src0 = M.dot(src0) #+ t0*np.ones(p0.T.shape)
        error_icp = np.sqrt((src0[0]-dst[0])**2 + (src0[1]-dst[1])**2 + (src0[2]-dst[2])**2)
        #print('Mean ICP error between transformed reconstruction and GPS data: {:.6f}, unit is meter.'.format(np.mean(error_icp)))

        error_sg = np.sqrt((sg[0]-dg[0])**2 + (sg[1]-dg[1])**2 + (sg[2]-dg[2])**2)
        #print('Mean selected ICP error between transformed reconstruction and GPS data: {:.6f}, unit is meter.'.format(np.mean(error_sg)))
        #src0 = src0.T
        #vis.show_tf_trajectory_3D(src0,sg,X_dst=dst,color=False,line=False)
        #vis.show_tf_trajectory_3D(src,sg,X_dst=dst,color=False,line=False)

        
        #vis.show_trajectory_3D(src0.T,dst.T)
        #error_icp = np.sqrt((traj_2[0]-traj_5[0])**2 + (traj_2[1]-traj_5[1])**2 + (traj_2[2]-traj_5[2])**2)
        ms = L2_error(dg[:,:3],  sg[:,:3])
        
        # del _
        diff = m_err - ms
        if 0 <= diff < 1e-10:
            #print('Previous error: {}\nCurrent Error: {}\nDifference: {}'.format(m_err, ms, diff))
            m_err = ms
            M_fin = M_t
            break
        src0 = sg
        m_err = ms
        M_t = M
        
        #if i%10 == 0:
            #print('L2 error after iteration {}: {}, diff: {}'.format(i, ms, diff))
    # X0, r, t = Q(p0, p2)
    # p0 = X0.dot(p0.T) # + t0*np.ones(p0.T.shape)
    # p0 = p0.T
    # m_err = L2_error(p2[:,:3],p0[:,:3])
    # print('L2 Error after registration: {}'.format(m_err))
    #X0[:3,:3] = r[:,:]
    #X0[:3,3] = t[:,0]
    #M[:3,:3] * scale.T
    return m_err, M_fin, sg,dg,scale #r, t.reshape((3,1)), p0


if __name__ == '__main__':

    # dir = 'point_cloud_registration'
    # filenames = ['pointcloud1.fuse', 'pointcloud2.fuse']
    # names = ['pointcloud1','pointcloud2']
    # pointcloud1 = readFile('{}/{}'.format(dir, filenames[0]), names[0])
    # pointcloud2 = readFile('{}/{}'.format(dir, filenames[1]), names[1])
    
    # print(pointcloud1.shape)
    # print(pointcloud2.shape)
    # i1 = None
    # i2 = None
    # if pointcloud1.shape[1] == 4:
    # 	# store available intensities separately
    # 	i1 = pointcloud1[:,3]
    # 	i2 = pointcloud2[:,3]
    # else:
    # 	# Create homogenous coordinates to redue time of transformations.
    # 	pointcloud1 = np.hstack((pointcloud1, np.ones((pointcloud1.shape[0],1))))
    # 	pointcloud2 = np.hstack((pointcloud2, np.ones((pointcloud2.shape[0],1))))

    # minPoints = min(pointcloud1.shape[0],pointcloud2.shape[0])

    # '''
    # Changing coordinate system from degrees to meters.
    # '''
    
    # # This segment is to store the sign information of the latitiudes and longitudes.
    # signLat = 1
    # if np.min(pointcloud1[:,0]) < 0:
    # 	signLat *= -1
    # signLong = 1
    # if np.min(pointcloud1[:,1]) < 0:
    # 	signLong *= -1


    # pc1 = np.zeros(pointcloud1.shape, dtype=pointcloud1.dtype)
    # pc2 = np.zeros(pointcloud2.shape, dtype=pointcloud2.dtype)

    # # This command converts the latitude coordinate to meters
    # pc1[:,0] = get_degree_to_meter(np.array([0.,0.]), np.hstack((pointcloud1[:,0].reshape((pc1.shape[0],1)), np.zeros((pc1.shape[0],1), dtype=np.float32))))[:]
    # # This command converts the longitude coordinate to meters
    # pc1[:,1] = get_degree_to_meter(np.array([0.,0.]), np.hstack((np.zeros((pc1.shape[0],1), dtype=np.float32), pointcloud1[:,1].reshape((pc1.shape[0],1)))))[:]
    # pc1[:,2] = pointcloud1[:,2]
    # pc1[:,3] = 1.

    # pc2[:,0] = get_degree_to_meter(np.array([0.,0.]), np.hstack((pointcloud2[:,0].reshape((pc2.shape[0],1)), np.zeros((pc2.shape[0],1), dtype=np.float32))))[:]
    # pc2[:,1] = get_degree_to_meter(np.array([0.,0.]), np.hstack((np.zeros((pc2.shape[0],1), dtype=np.float32), pointcloud2[:,1].reshape((pc2.shape[0],1)))))[:]
    # pc2[:,2] = pointcloud2[:,2]
    # pc2[:,3] = 1.

    # minX = min(np.min(pc1[:,0]), np.min(pc2[:,0]))
    # minY = min(np.min(pc1[:,1]), np.min(pc2[:,1]))

    # cs_Mat = np.eye(4,dtype=pointcloud1.dtype)
    # cs_Mat[0,3] = -minX
    # cs_Mat[1,3] = -minY
    # rev_cs_Mat = np.linalg.inv(cs_Mat)
    # print('Coordinate system conversion matrix:')
    # print(cs_Mat)
    # print('Matrix to get back original coordinates:')
    # print(rev_cs_Mat)

    # pc1 = cs_Mat.dot(pc1.T)
    # pc1 = pc1.T
    # pc2 = cs_Mat.dot(pc2.T)
    # pc2 = pc2.T
    # # pc1 = pointcloud1
    # # pc2 = pointcloud2

    # err = None
    # initRot = ROTNODE()
    # initTrans = TRANSNODE()

    # initRot.a = -np.pi
    # initRot.b = -np.pi
    # initRot.c = -np.pi
    # initRot.w = 2*np.pi
    # initRot.l = 0

    # initTrans.x = -0.5
    # initTrans.y = -0.5
    # initTrans.z = -0.5
    # initTrans.w = 1

    # initTrans.lb = 0.
    # initRot.lb = 0.

    # print()
    # cur = dt.now()
    # # minPoints = 10000

    # # We want the number of points in the model and data point cloud to be same

    '''
    The command below is to run ICP only.
    '''
    
    err, X0, R, T, final_p1 = ICP(pc1[:minPoints,:], pc2[:minPoints,:], KDTree(pc2[:minPoints,:]))
    
