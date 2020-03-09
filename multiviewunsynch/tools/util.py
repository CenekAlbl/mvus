# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math
import numpy as np
from scipy import interpolate
from thirdparty import transformation
from tools import ransac
# from numba import jit


def mapminmax(x,ymin,ymax):
    return (ymax-ymin)*(x-min(x))/(max(x)-min(x)) + ymin


def rotation(x,y,z):
    x,y,z = x/180*math.pi, y/180*math.pi, z/180*math.pi

    Rx = np.array([[1,0,0],[0,math.cos(x),-math.sin(x)],[0,math.sin(x),math.cos(x)]])
    Ry = np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
    Rz = np.array([[math.cos(z),-math.sin(z),0],[math.sin(z),math.cos(z),0],[0,0,1]])

    return np.dot(np.dot(Rz,Ry),Rx)


def rotation_decompose(R):
    # x = np.arctan2(R[2,1],R[2,2])
    # y = np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2))
    # z = np.arctan2(R[1,0],R[0,0])

    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    assert(n < 1e-6)

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # return x,y,z
    return x*180/math.pi, y*180/math.pi, z*180/math.pi


def homogeneous(x):
    return np.vstack((x,np.ones(x.shape[1])))

# @jit
def find_intervals(x,gap=5,idx=False):
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

# @jit
def sampling(x,interval,belong=False):
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


def match_overlap(x,y):
    '''
    Given two inputs in the same timeline (global), return the parts of them which are temporally overlapped

    Important: it's assumed that x has a higher frequency (fps) so that points are interpolated in y
    '''

    interval = find_intervals(y[0])
    x_s, _ = sampling(x, interval)

    tck, u = interpolate.splprep(y[1:],u=y[0],s=0,k=3)
    y_s = np.asarray(interpolate.splev(x_s[0],tck))
    y_s = np.vstack((x_s[0],y_s))

    assert (x_s[0] == y_s[0]).all(), 'Both outputs should have the same timestamps'

    return x_s, y_s

        
def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


if __name__ == "__main__":
    R = rotation(0.38,-176.3,100)
    x,y,z = rotation_decompose(R)
    T = rotation(x,y,z)

    print('Finished')