import numpy as np
import pickle
from scipy.optimize import least_squares
from tools import ransac, util
import tools.visualization as vis
from thirdparty import transformation


def estimate_M(data,param=None):
    reconst = data[:3]
    gt = data[3:]
    M = transformation.affine_matrix_from_points(reconst,gt,shear=False,scale=True)
    
    return M.ravel()


def error_M(model,data,param=None):
    reconst = data[:3]
    gt = data[3:]
    M = model.reshape(4,4)

    tran = np.dot(M,util.homogeneous(reconst))
    tran /= tran[-1]
    
    return np.sqrt((gt[0]-tran[0])**2 + (gt[1]-tran[1])**2 + (gt[2]-tran[2])**2)


def optimize(alpha, beta, flight, gt):


    def error_fn(model,output=False):
        alpha, beta = model[0], model[1]
        if gt.shape[0] == 3:
            t_gt = alpha * np.arange(gt.shape[1]) + beta
        else:
            t_gt = alpha * (gt[0]-gt[0,0]) + beta
        _, idx = util.sampling(t_gt, flight.spline['int'])
        gt_part = gt[-3:,idx]
        t_part = t_gt[idx]
        traj = flight.spline_to_traj(t=t_part)

        data = np.vstack((traj[1:],gt_part))
        error = np.zeros(gt.shape[1],dtype=float)

        # result = ransac.vanillaRansac(estimate_M,error_M,data,3,2,100)
        # error[idx] = error_M(result['model'],data)

        M = transformation.affine_matrix_from_points(traj[1:],gt_part,shear=False,scale=True)
        error[idx] = error_M(M.ravel(),data)
        
        if output:
            traj_tran = np.dot(M,util.homogeneous(traj[1:]))
            traj_tran /= traj_tran[-1]
            return np.vstack((traj[0],traj_tran[:3])), gt_part, M, error[idx]
        else:
            return error


    model = np.array([alpha, beta])

    fn = lambda x: error_fn(x)
    ls = least_squares(fn,model,loss='cauchy',f_scale=1)

    res = error_fn(np.array([ls.x[0],ls.x[1]]),output=True)

    return ls, res


def align_gt(flight, f_gt, gt_path, visualize=False):

    if not len(gt_path):
        print('No ground truth data provided\n')
        return
    else:
        try:
            gt_ori = np.loadtxt(gt_path)
        except:
            print('Ground truth not correctly loaded')
            return

    if gt_ori.shape[0] == 3 or gt_ori.shape[0] == 4:
        pass
    elif gt_ori.shape[1] == 3 or gt_ori.shape[1] == 4:
        gt_ori = gt_ori.T
    else:
        raise Exception('Ground truth data have an invalid shape')

    # Pre-processing
    f_reconst = flight.cameras[flight.settings['ref_cam']].fps
    alpha = f_reconst/f_gt

    reconst = flight.spline_to_traj(sampling_rate=alpha)
    t0 = reconst[0,0]
    reconst = np.vstack(((reconst[0]-t0)/alpha,reconst[1:]))
    if gt_ori.shape[0] == 3:
        gt = np.vstack((np.arange(len(gt_ori[0])),gt_ori))
    else:
        gt = gt_ori

    # Coarse search
    error_min = np.inf
    for i in range(int(gt[0,-1]-reconst[0,-1])):
        reconst_i = np.vstack((reconst[0]+i,reconst[1:]))
        p1, p2 = util.match_overlap(reconst_i, gt)
        M = transformation.affine_matrix_from_points(p1[1:], p2[1:], shear=False, scale=True)

        tran = np.dot(M, util.homogeneous(p1[1:]))
        tran /= tran[-1]
        error_all = np.sqrt((p2[1]-tran[0])**2 + (p2[2]-tran[1])**2 + (p2[3]-tran[2])**2)
        error = np.sum(error_all**2) # np.mean(error_all)
        if error < error_min:
            error_min = error
            error_coarse = error_all
            j = i
    beta = t0-alpha*j

    # Fine optimization
    ls, res = optimize(alpha,beta,flight,gt_ori)

    # Remove outliers by relative thresholding
    thres = 10
    error_ = res[3]
    idx = error_ <= thres*np.mean(error_)
    reconst_, gt_, error_ = res[0][:,idx], res[1][:,idx], error_[idx]

    # Result
    out = {'align_param':ls.x, 'reconst_tran':reconst_, 'gt':gt_, 'tran_matrix':res[2], 'error':error_}
    print('The mean error (distance) is {:.5f} meter\n'.format(np.mean(out['error'])))
    print('The median error (distance) is {:.5f} meter\n'.format(np.median(out['error'])))
    print('The max error (distance) is {:.5f} meter\n'.format(np.max(out['error'])))
    print('The min error (distance) is {:.5f} meter\n'.format(np.min(out['error'])))

    print(ls.x)

    if visualize:
        # Compare the trajectories
        vis.show_trajectory_3D(out['reconst_tran'][1:], out['gt'], line=False, title='Reconstruction(left) vs Ground Truth(right)')

        # Error histogram
        vis.error_hist(out['error'])

        # Error over the trajectory
        vis.error_traj(out['reconst_tran'][1:], out['error'])

    return out


if __name__ == "__main__":

    # Load the reconstructed trajectory
    reconst_path = ''
    with open(reconst_path, 'rb') as file:
        flight = pickle.load(file)

    # Load the ground truth data
    gt_path = ''

    out = align_gt(flight, 5, gt_path, visualize=True)

    print('Finish!')
