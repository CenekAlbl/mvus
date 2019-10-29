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
    with open('./data/paper/fixposition/trajectory/flight_5cam_1.pkl', 'rb') as file:
        flight = pickle.load(file)
    traj = flight.traj

    # Load the GPS data
    gps_ori = np.loadtxt('./data/paper/fixposition/GT_position/GT_ENU.txt').T

    # Find the longest continues part of the reconstruction
    part = np.where(traj[0,1:]-traj[0,:-1] != 1)[0]
    part = np.insert(part,[0,len(part)],[-1,traj.shape[1]-1])
    part_max_id = np.argmax(part[1:]-part[:-1])
    id_start = part[part_max_id]+1
    id_end = part[part_max_id+1]
    traj_cont = traj[:,id_start:id_end+1]

    # Set matching parameters
    ratio = 6
    step = 0.1
    k = 781
    error_min = np.inf
    num_GPS = int(traj_cont.shape[1] / ratio) - 20

    gps_part = gps_ori[:,k:k+num_GPS]
    tck, u = splprep(traj_cont[1:],u=traj_cont[0],s=0)
    idx = np.arange(int(u[0]),int(u[0])+num_GPS*6)

    # Show the initial part from GPS und check if it corresponds to the recontruction
    vis.show_trajectory_3D(traj_cont[1:],gps_part)

    # Using Ransac to estimate transformation parameters in each small step
    start = datetime.now()

    while idx[-1] < traj_cont[0,-1]:
        s = splev(idx[::ratio],tck)
        traj_down = np.array([s[0],s[1],s[2]])
        assert traj_down.shape[1] == num_GPS, 'The number of sampled points should be the same as that of the GPS data'

        # Ransac
        data = np.vstack((traj_down,gps_part))
        result = ransac.vanillaRansac(estimate_M,error_M,data,3,0.5,300)
        # result = ransac.loRansacSimple(estimate_M,error_M,data,3,0.5,300)

        # Get errors after Ransac
        M = result['model'].reshape(4,4)
        traj_tran = np.dot(M,util.homogeneous(traj_down))
        traj_tran /= traj_tran[-1]
        error = np.sqrt((gps_part[0]-traj_tran[0])**2 + (gps_part[1]-traj_tran[1])**2 + (gps_part[2]-traj_tran[2])**2)

        # Check if this step is the best so far
        if np.mean(error) < error_min:
            error_min = np.mean(error)
            idx_ransac = idx[::ratio]
            M_ransac = M
            result_ransac = result
            step_diff = idx[0]-u[0]

        idx = idx + step

    # Apply the estimated transformation so that the reconstruction is transformed in ENU system
    s = splev(idx_ransac,tck)
    traj_down = np.array([s[0],s[1],s[2]])
    traj_tran = np.dot(M_ransac,util.homogeneous(traj_down))
    traj_tran /= traj_tran[-1]
    error = np.sqrt((gps_part[0]-traj_tran[0])**2 + (gps_part[1]-traj_tran[1])**2 + (gps_part[2]-traj_tran[2])**2)

    # Show results
    print('\nTime: {}\n'.format(datetime.now()-start))
    print('The mean error (distance) of the PART is {:.3f} meter'.format(np.mean(error)))
    vis.show_trajectory_3D(traj_tran,gps_part)

    #  Apply the estimated transformation to the entire reconstruction
    idx_gps = (traj[0]-idx_ransac[0])/6 +k
    idx_gps = np.unique(idx_gps.astype(int))
    traj_tran_all = np.dot(M_ransac,util.homogeneous(traj[1:]))
    traj_tran_all /= traj_tran_all[-1]

    idx_sample = (idx_gps - k)*6 + idx_ransac[0]
    tck, u = splprep(traj[1:],u=traj[0],s=0)
    s = splev(idx_sample,tck)
    traj_sample = np.array([s[0],s[1],s[2]])
    traj_tran_sample = np.dot(M_ransac,util.homogeneous(traj_sample))
    traj_tran_sample /= traj_tran_sample[-1]
    error_all = np.sqrt((gps_ori[0,idx_gps]-traj_tran_sample[0])**2 + (gps_ori[1,idx_gps]-traj_tran_sample[1])**2 + (gps_ori[2,idx_gps]-traj_tran_sample[2])**2)
    print('The mean error (distance of the ENTIRE is {:.3f} meter'.format(np.mean(error_all)))

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter3D(traj_tran_all[0],traj_tran_all[1],traj_tran_all[2],c='r',s=100)
    ax.scatter3D(gps_ori[0,idx_gps],gps_ori[1,idx_gps],gps_ori[2,idx_gps],c='b',s=100)
    ax.set_xlabel('East',fontsize=20)
    ax.set_ylabel('North',fontsize=20)
    ax.set_zlabel('Up',fontsize=20)
    plt.show()
    vis.show_trajectory_3D(traj_tran_all,gps_ori[:,idx_gps],line=False,color=False)


    print('Finish!')
