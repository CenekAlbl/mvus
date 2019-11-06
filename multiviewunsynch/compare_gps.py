# This script estimates a transformation between gps and reconstructed trajectory

import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import splprep, splev
from scipy.optimize import least_squares
import util
import visualization as vis
import transformation
import ransac


def estimate_M(data,param=None):
    reconst = data[:3]
    gps = data[3:]
    M = transformation.affine_matrix_from_points(reconst,gps,shear=False,scale=True)
    
    return M.ravel()


def error_M(model,data,param=None):
    reconst = data[:3]
    gps = data[3:]
    M = model.reshape(4,4)

    tran = np.dot(M,util.homogeneous(reconst))
    tran /= tran[-1]
    
    return np.sqrt((gps[0]-tran[0])**2 + (gps[1]-tran[1])**2 + (gps[2]-tran[2])**2)


def optimize(alpha, beta, flight, gps):

    def error_fn(model,output=False):
        alpha, beta = model[0], model[1]
        t_gps = alpha * np.arange(gps.shape[1]) + beta
        _, idx = flight.sampling(t_gps, flight.spline[1])
        gps_part = gps[:,idx]
        t_part = t_gps[idx]
        traj = flight.spline_to_traj(t=t_part)

        data = np.vstack((traj[1:],gps_part))
        error = np.zeros(gps.shape[1],dtype=float)

        # result = ransac.vanillaRansac(estimate_M,error_M,data,3,2,100)
        # error[idx] = error_M(result['model'],data)

        M = transformation.affine_matrix_from_points(traj[1:],gps_part,shear=False,scale=True)
        error[idx] = error_M(M.ravel(),data)
        
        if output:
            return traj, gps_part, M, error
        else:
            return error


    model = np.array([alpha, beta])

    fn = lambda x: error_fn(x)
    ls = least_squares(fn,model)

    traj, gps_part, M, error = error_fn(np.array([ls.x[0],ls.x[1]]),output=True)

    return ls, [traj, gps_part, M, error]


if __name__ == "__main__":

    # Load the reconstructed trajectory
    with open('./data/paper/fixposition/trajectory/flight_spline_2.pkl', 'rb') as file:
        flight = pickle.load(file)

    # Load the GPS data
    gps_ori = np.loadtxt('./data/paper/fixposition/GT_position/GT_ENU.txt').T

    # Set parameters
    f_gps = 5
    f_spline = 30
    alpha = f_spline / f_gps
    beta = -1360
    error_min = np.inf

    # Optimization
    ls, result = optimize(alpha,beta,flight,gps_ori)

    # Apply the estimated transformation and show results
    alpha, beta = ls.x[0], ls.x[1]
    traj, gps_part, M, error = result[0], result[1], result[2], result[3],

    traj_tran = np.dot(M,util.homogeneous(traj[1:]))
    traj_tran /= traj_tran[-1]
    print('The mean error (distance) is {:.3f} meter\n'.format(np.mean(error)))
    vis.show_trajectory_3D(traj_tran, gps_part, line=False)

    print('Finish!')
