# This script estimates a transformation between gps and reconstructed trajectory

import numpy as np
import pickle
from scipy.optimize import least_squares
from tools import ransac, util
import tools.visualization as vis
from thirdparty import transformation


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
        try:
            _, idx = util.sampling(t_gps, flight.spline[1])
        except:
            _, idx = util.sampling(t_gps, flight.spline['int'])
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
            gps_idx = np.arange(len(idx))
            return traj, gps_part, M, error, gps_idx[idx]
        else:
            return error


    model = np.array([alpha, beta])

    fn = lambda x: error_fn(x)
    ls = least_squares(fn,model)

    res = error_fn(np.array([ls.x[0],ls.x[1]]),output=True)

    return ls, res


if __name__ == "__main__":

    # Load the reconstructed trajectory
    reconst_path = 'data/fixposition/trajectory/flight_ds2.pkl'
    with open(reconst_path, 'rb') as file:
        flight = pickle.load(file)

    # Load the GPS data
    gps_path = 'data/fixposition/Raw_gps/GT_ENU.txt'
    gps_ori = np.loadtxt(gps_path).T


    '''-----------------Transformation estimation-----------------'''
    # Set parameters
    if flight.gps['alpha'] and flight.gps['beta']:
        alpha = flight.gps['alpha']
        beta = flight.gps['beta']
    else:
        # Please enter the numbers manually if thers's no prior values
        f_gps = 5
        f_spline = 29.727612      # fixposition: 29.727612    thesis1: 29.970030      thesis2: 29.970030
        alpha = f_spline / f_gps
        beta = -1290         # fixposition: -1290        thesis1: -19700         thesis2: -4720
    error_min = np.inf

    # Optimization
    ls, result = optimize(alpha,beta,flight,gps_ori)

    # Apply the estimated transformation and show results
    alpha, beta = ls.x[0], ls.x[1]
    traj, gps_part, M, gps_idx = result[0], result[1], result[2], result[4]

    traj_tran = np.dot(M,util.homogeneous(traj[1:]))
    traj_tran /= traj_tran[-1]
    error = np.sqrt(np.sum((gps_part-traj_tran[:3])**2,axis=0))
    print('The mean error (distance) is {:.5f} meter\n'.format(np.mean(error)))


    '''-----------------Visualization-----------------'''
    # Compare the trajectories
    vis.show_trajectory_3D(traj_tran, gps_part, line=False)

    # Error histogram
    vis.error_hist(error)

    # Error over the trajectory
    vis.error_traj(traj_tran[:3], error)

    # Reprojection to 2D
    interval = np.array([[0],[149000]])
    flight.plot_reprojection(interval,match=True)

    # save the comparison result
    flight.gps = {'alpha':alpha, 'beta':beta, 'gps':gps_part, 'traj':traj_tran[:3], 'timestamp':traj[0], 'M':M, 'gps_idx':gps_idx}
    with open(reconst_path,'wb') as f:
        pickle.dump(flight, f) 

    print('Finish!')
