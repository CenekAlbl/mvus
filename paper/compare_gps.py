# This script estimates a transformation between gps and reconstructed trajectory

import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import splprep, splev
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


if __name__ == "__main__":

    # Load the reconstructed trajectory
    with open('./data/paper/fixposition/trajectory/flight_spline_2.pkl', 'rb') as file:
        flight = pickle.load(file)

    # Load the GPS data
    gps_ori = np.loadtxt('./data/paper/fixposition/GT_position/GT_ENU.txt').T

    # Set parameters
    f_gps = 5
    f_spline = 29.727612
    alpha = f_spline / f_gps
    beta = -1360
    error_min = np.inf

    # Using Ransac to estimate transformation parameters in each small step
    start = datetime.now()

    beta_list = np.arange(beta-10,beta+10,0.1)
    for beta in beta_list:
        t_gps = alpha * np.arange(gps_ori.shape[1]) + beta
        _, idx = flight.sampling(t_gps, flight.spline[1])
        gps_part = gps_ori[:,idx]
        t_part = t_gps[idx]
        traj = flight.spline_to_traj(t=t_part)

        # Ransac
        data = np.vstack((traj[1:],gps_part))
        result = ransac.vanillaRansac(estimate_M,error_M,data,3,2,100)

        # Get errors after Ransac
        error = error_M(result['model'],data)

        # Check if this step is the best so far
        if np.mean(error) < error_min:
            error_min = np.mean(error)
            error_ransac = error
            beta_ransac = beta
            M_ransac = result['model'].reshape(4,4)
            result_ransac = result
            traj_ransac = traj
            gps_ransac = gps_part
    print('\nTotal time: {}\n'.format(datetime.now()-start))

    # Apply the estimated transformation and show results
    traj_tran = np.dot(M_ransac,util.homogeneous(traj_ransac[1:]))
    traj_tran /= traj_tran[-1]
    print('The mean error (distance) is {:.3f} meter'.format(np.mean(error_ransac)))
    vis.show_trajectory_3D(traj_tran, gps_ransac, line=False)

    # # Show results
    # print('\nTime: {}\n'.format(datetime.now()-start))
    # print('The mean error (distance) of the PART is {:.3f} meter'.format(np.mean(error)))
    # vis.show_trajectory_3D(traj_tran,gps_part)

    # #  Apply the estimated transformation to the entire reconstruction
    # idx_gps = (traj[0]-idx_ransac[0])/6 +k
    # idx_gps = np.unique(idx_gps.astype(int))
    # traj_tran_all = np.dot(M_ransac,util.homogeneous(traj[1:]))
    # traj_tran_all /= traj_tran_all[-1]

    # idx_sample = (idx_gps - k)*6 + idx_ransac[0]
    # tck, u = splprep(traj[1:],u=traj[0],s=0)
    # s = splev(idx_sample,tck)
    # traj_sample = np.array([s[0],s[1],s[2]])
    # traj_tran_sample = np.dot(M_ransac,util.homogeneous(traj_sample))
    # traj_tran_sample /= traj_tran_sample[-1]
    # error_all = np.sqrt((gps_ori[0,idx_gps]-traj_tran_sample[0])**2 + (gps_ori[1,idx_gps]-traj_tran_sample[1])**2 + (gps_ori[2,idx_gps]-traj_tran_sample[2])**2)
    # print('The mean error (distance of the ENTIRE is {:.3f} meter'.format(np.mean(error_all)))

    # fig = plt.figure(figsize=(20, 15))
    # ax = fig.add_subplot(111,projection='3d')
    # ax.scatter3D(traj_tran_all[0],traj_tran_all[1],traj_tran_all[2],c='r',s=100)
    # ax.scatter3D(gps_ori[0,idx_gps],gps_ori[1,idx_gps],gps_ori[2,idx_gps],c='b',s=100)
    # ax.set_xlabel('East',fontsize=20)
    # ax.set_ylabel('North',fontsize=20)
    # ax.set_zlabel('Up',fontsize=20)
    # plt.show()
    # vis.show_trajectory_3D(traj_tran_all,gps_ori[:,idx_gps],line=False,color=False)


    print('Finish!')
