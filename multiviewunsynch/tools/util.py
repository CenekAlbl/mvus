import math
import numpy as np
from reconstruction import common
from scipy.interpolate import UnivariateSpline
from thirdparty import transformation
import pymap3d as pm
from tools import ransac


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


def spline_fitting(x,t,t0=[],k=1,s=0,return_object=False):
    '''
    This function reads an array of samples (x) and return interpolated values at given positions (t)
    '''
    if not len(t0):
        t0 = np.arange(x.shape[0])
    spl = UnivariateSpline(t0, x, k=k, s=s)

    if return_object:
        return spl
    else:
        return spl(t)


def homogeneous(x):
    return np.vstack((x,np.ones(x.shape[1])))


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

def gps_to_enu(gps_xyz,out_file):
        # Load ground truth 
        gt = np.loadtxt(gps_xyz).T

        ell_wgs84 = pm.Ellipsoid('wgs84')

        gt_ll = np.vstack(pm.ecef2geodetic(gt[0],gt[1],gt[2],ell=ell_wgs84))
        gt_enu = np.vstack(pm.geodetic2enu(gt_ll[0],gt_ll[1],gt_ll[2],gt_ll[0][-10],gt_ll[1][-10],gt_ll[2][-10],ell=ell_wgs84))

        np.savetxt(out_file, gt_enu.T,delimiter='  ') 

def sim_tran(src, dst, thres=0.5):

    def model_fn(data,param):
        x, y = data[:3], data[3:]
        M = transformation.affine_matrix_from_points(x,y,shear=False)
        return M.ravel()
    
    def error_fn(model,data,param):
        x, y = data[:3], data[3:]
        M = model.reshape((4,4))
        y_t = np.dot(M,homogeneous(x))
        y_t /= y_t[-1]
        return np.sqrt((y[0]-y_t[0])**2+(y[1]-y_t[1])**2+(y[2]-y_t[2])**2)

    Data = np.vstack((src,dst))
    return ransac.vanillaRansac(model_fn,error_fn,Data,10,thres,500)



if __name__ == "__main__":
    R = rotation(0.38,-176.3,100)
    x,y,z = rotation_decompose(R)
    T = rotation(x,y,z)

    print('Finished')