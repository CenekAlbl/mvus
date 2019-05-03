import numpy as np
import cv2
import util
import pickle
import epipolar as ep
import synchronization
import visualization as vis
from datetime import datetime
from scipy.interpolate import UnivariateSpline

'''
This script tests the estimation of fundamental matrix F and 3D reconstruction of the real trajectory data

1. with and without considering synchronization

2. with and without spline fitting
'''

start=datetime.now()

# Load data
traj_1 = np.loadtxt('data/video_1_output.txt',skiprows=1,dtype=np.int32)
traj_2 = np.loadtxt('data/video_2_output.txt',skiprows=1,dtype=np.int32)

# Spline fitting
k, s = 1, 0
t1 = np.arange(traj_1.shape[0])
t2 = np.arange(traj_2.shape[0])
spl_1_x = UnivariateSpline(t1, traj_1[:,1], k=k, s=s)
spl_1_y = UnivariateSpline(t1, traj_1[:,2], k=k, s=s)
spl_2_x = UnivariateSpline(t2, traj_2[:,1], k=k, s=s)
spl_2_y = UnivariateSpline(t2, traj_2[:,2], k=k, s=s)

traj_1[:,1] = spl_1_x(t1)
traj_1[:,2] = spl_1_y(t1)
traj_2[:,1] = spl_2_x(t2)
traj_2[:,2] = spl_2_y(t2)

print('Degree of spline:{}, smoothing factor:{}\n'.format(k,s))
print('Numbers of control points for traj 1, x:{}, y:{}\n'.format(len(spl_1_x.get_knots()),len(spl_1_y.get_knots())))
print('Numbers of control points for traj 2, x:{}, y:{}\n'.format(len(spl_2_x.get_knots()),len(spl_2_y.get_knots())))
print('Residual of these splines: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]\n'.format(
    spl_1_x.get_residual(),spl_1_y.get_residual(),spl_2_x.get_residual(),spl_2_y.get_residual()))

# Load calibration matrix
calibration = open('data/calibration.pickle','rb')
K1 = np.asarray(pickle.load(calibration)["intrinsic_matrix"])
K2 = K1

# Define shifting of trajectory
start_1 = 153
start_2 = 71
num_traj = 1500
shift_range = np.arange(-20,21,2)

# iterate over all shifts
results = {'shift':[], 'Beta':[], 'X1':[], 'X2':[], 'P2_1':[], 'P2_2':[], 'inlier1':[], 'inlier2':[], 'k':k, 's':s}
it = 0
while it < len(shift_range):
    shift = shift_range[it]
    print('\nCurrent shift: {}\n'.format(shift))

    x1 = np.vstack((traj_1[start_1+shift:start_1+shift+num_traj,1:].T, np.ones(num_traj)))
    x2 = np.vstack((traj_2[start_2:start_2+num_traj,1:].T, np.ones(num_traj)))

    # vis.show_trajectory_2D(x1,x2,title='Degree of spline:{}, smoothing factor:{}'.format(k,s))

    '''Compute fundamental Matrix'''
    # Without synchronization
    F1, mask = ep.computeFundamentalMat(x1, x2, error=5)
    inliers1 = sum(mask.T[0]==1) / num_traj

    # With synchronization
    param = {'k':1, 's':0}
    # Brute-force
    # beta, F2, inliers2 = synchronization.search_sync(x1,x2,param,d_min=-6,d_max=6,threshold1=5,threshold2=5,maxiter=300,loRansac=True)
    # iterative
    beta, F2, inliers2 = synchronization.iter_sync(x1,x2,param,p_max=5,threshold=5,maxiter=500,loRansac=False)
    F2 = F2.reshape((3,3))

    '''Triangulation'''
    # Without Beta
    E1 = np.dot(np.dot(K2.T,F1),K1)
    X1, P2_1 = ep.triangulate_from_E(E1,K1,K2,x1,x2)

    # Show 3D results
    # vis.show_trajectory_3D(X1)

    # With Beta
    E2 = np.dot(np.dot(K2.T,F2),K1)
    x2_shift = synchronization.shift_trajectory(x2,beta,k=param['k'],s=param['s'])
    if beta >= 1:
        s2 = x2_shift[:,:-int(beta)]
        s1 = x1[:,:-int(beta)]
    else:
        s2 = x2_shift[:,-int(beta):]
        s1 = x1[:,-int(beta):]
    X2, P2_2 = ep.triangulate_from_E(E1,K1,K2,s1,s2)

    # Show 3D results
    # vis.show_trajectory_3D(X1,X2)

    # Save results
    results['shift'].append(shift)
    results['Beta'].append(beta)
    results['X1'].append(X1)
    results['X2'].append(X2)
    results['P2_1'].append(P2_1)
    results['P2_2'].append(P2_2)
    results['inlier1'].append(inliers1)
    results['inlier2'].append(inliers2)
    it += 1

    print('\nTime: ',datetime.now()-start)

# Save parameters
results['k'], results['s'] = k, s
results['error'], results['threshold'], results['maxiter'], results['loRansac'] = 5, 5, 500, False

file = open('data/test_trajectory.pickle', 'wb')
pickle.dump(results, file)
file.close()

print('\nfinished\n')
    