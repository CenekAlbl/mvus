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
            _, idx = flight.sampling(t_gps, flight.spline[1])
        except:
            _, idx = flight.sampling(t_gps, flight.spline['int'])
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
    data_path = './data/paper/fixposition/trajectory/flight_rs_false.pkl'
    with open(data_path, 'rb') as file:
        flight = pickle.load(file)

    # Load the GPS data
    gps_ori = np.loadtxt('./data/paper/fixposition/GT_position/GT_ENU.txt').T

    # error = np.sqrt(np.sum((flight.gps['gps']-flight.gps['traj'])**2,axis=0))
    # vis.error_traj(flight.gps['traj'], error,size=50,colormap='Wistia')

    '''-----------------Transformation estimation-----------------'''
    # Set parameters
    f_gps = 5
    f_spline = 29.727612      # fixposition: 29.727612    thesis1: 29.970030      thesis2: 29.970030
    alpha = f_spline / f_gps
    beta = -1290         # fixposition: -1290        thesis1: -19700         thesis2: -4720
    error_min = np.inf

    # Optimization
    ls, result = optimize(alpha,beta,flight,gps_ori)

    # Apply the estimated transformation and show results
    alpha, beta = ls.x[0], ls.x[1]
    traj, gps_part, M = result[0], result[1], result[2]

    traj_tran = np.dot(M,util.homogeneous(traj[1:]))
    traj_tran /= traj_tran[-1]
    error = np.sqrt(np.sum((gps_part-traj_tran[:3])**2,axis=0))
    print('The mean error (distance) is {:.5f} meter\n'.format(np.mean(error)))

    # # Print timestamps of large errors
    # thres = 1
    # t_large_trans = traj[0,error>thres].astype(int)
    # print('Global timestamps for large errors after comparison with GPS:  ', t_large_trans)
    # for i in range(flight.numCam):
    #     error_cam_i = flight.error_cam(i,'each')
    #     error_xy = np.split(error_cam_i,2)
    #     error_cam_i = np.sqrt(error_xy[0]**2 + error_xy[1]**2)
    #     idx_large = np.argsort(error_cam_i)[-len(t_large_trans):]
    #     t_large_reconst = flight.detections_global[i][0,idx_large].astype(int)
    #     print('Global timestamps for large errors from camera{}:  '.format(i), t_large_reconst)


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
    flight.gps = {'alpha':alpha, 'beta':beta, 'gps':gps_part, 'traj':traj_tran[:3], 'timestamp':traj[0], 'M':M}
    with open(data_path,'wb') as f:
        pickle.dump(flight, f) 

    print('Finish!')
