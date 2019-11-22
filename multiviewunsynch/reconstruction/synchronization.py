import pickle
import numpy as np
import scipy.linalg
import reconstruction.epipolar as ep
from tools import ransac, video
import cv2
import argparse
from datetime import datetime
import tools.visualization as vis
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline


def shift_trajectory(x,beta,k=3,s=-1):
    '''
    Shift trajectory with given frames

    The shifted trajectory will have the same size as the original one
    '''

    num = x.shape[1]
    t = np.arange(num, dtype=float)
    if s==-1:
        spl_x = UnivariateSpline(t, x[0], k=k)
        spl_y = UnivariateSpline(t, x[1], k=k)
    else:
        spl_x = UnivariateSpline(t, x[0], k=k, s=s)
        spl_y = UnivariateSpline(t, x[1], k=k, s=s)

    return np.vstack((np.vstack((spl_x(t+beta),spl_y(t+beta))),np.ones(num)))


def velocity_vector(x,d):
    '''
    Compute the approximated tangent vector over next d frames

    if d>0, the last d frames are ignored

    if d<0, the first -d frames are ignored
    '''

    if d>0:
        return x[:,d:]-x[:,:-d]
    else:
        return x[:,:d]-x[:,-d:]


def compute_beta(data):
    '''
    Read 9 correspondences and the velocity vector, return the possible value of beta

    x1 is fixed, shifting x2 with beta frames will fit the virtual correspondence
    '''

    # Check input
    if data.shape[1]!=9:
        raise ValueError('Number of input points must be 9!')

    # decompose input data
    s1 = data[:3,:]
    s2 = data[3:6,:]
    ds = data[6:,:]

    # Create design matrices A1, A2
    A1 = np.array([s1[0]*s2[0],s1[0]*s2[1],s1[0],
                    s1[1]*s2[0],s1[1]*s2[1],s1[1],
                    s2[0],s2[1],np.ones(9)]).T

    A2 = np.array([s1[0]*ds[0],s1[0]*ds[1],np.zeros(9),
                s1[1]*ds[0],s1[1]*ds[1],np.zeros(9),
                ds[0],ds[1],np.zeros(9)]).T

    # Compute eigenvalue
    w = scipy.linalg.eigvals(A1,A2)

    return -w


def compute_beta_fundamental(data,param):
    '''
    This function reads data of 9 points and return a list of possible solution for Beta and F

    It can be called by the Ransac function as a function handle 
    '''

    # compute possible beta
    w = compute_beta(data)
    w[np.iscomplex(w)] = np.inf
    beta = w.real[np.isfinite(w)]

    # decompose input data
    s1 = data[:3,:]
    s2 = data[3:6,:]
    ds = data[6:,:]

    # compute fundamental matrix F using shifted data
    M = np.zeros((len(beta),10))
    for i in range(len(beta)):
        x2 = s2 + beta[i]*ds
        # F = ep.compute_fundamental(s1,x2)
        F, mask = cv2.findFundamentalMat(s1[:2].T,x2[:2].T,method=cv2.FM_8POINT)

        if len(np.ravel(F)) == 9:
            M[i] = np.append(np.ravel(F),np.array([beta[i]*param['d']]))
        else:
            M[i] = np.append(np.ones(9),np.array([beta[i]*param['d']]))

    return M


def compute_beta_fundamental_Ransac(x1,x2,param,threshold=10e-4,maxiter=500,verbose=False,loRansac=False):
    '''
    This function calls the Ransac for computing Beta and F and returns the best model and inliers.
    '''

    # Load parameter
    d = param['d']

    # Create the approximated velocity vectors
    ds = velocity_vector(x2,d)
    if d>0:
        data = np.append(np.append(x1[:,:-d],x2[:,:-d],axis=0),ds,axis=0)
    else:
        data = np.append(np.append(x1[:,-d:],x2[:,-d:],axis=0),ds,axis=0)

    if loRansac:
        return ransac.loRansacSimple(compute_beta_fundamental,error_beta_fundamental,data,9,threshold,maxiter,param,verbose=verbose)
    else:
        return ransac.vanillaRansac(compute_beta_fundamental,error_beta_fundamental,data,9,threshold,maxiter,param,verbose=verbose)


def error_beta_fundamental(M,data,param):
    '''
    This function computes the Sampson error given Beta and F

    The shifting of trajectory will be the same as input data
    '''

    # Load parameter
    k = param['k']
    s = param['s']

    # decompose model
    F = M[:9].reshape((3,3))
    beta = M[9]

    # decompose input data
    x1 = data[:3,:]
    if data.shape[1] > 5:
        x1 = shift_trajectory(x1,0,k=k,s=s)
        x2 = shift_trajectory(data[3:6,:],beta,k=k,s=s)
        return ep.Sampson_error(x1,x2,F)
    else:
        return np.ones(data.shape[1])*10000


def verify_beta(x1,x2,beta,param,show=True):
    '''
    This is a simple function to show the functionality of computing Beta and F
    '''

    # Load parameter
    d = param['d']
    k = param['k']
    s = param['s']

    # Create the approximated velocity vectors
    ds = velocity_vector(x2,d)
    if beta>1:
        s1 = shift_trajectory(x1,beta,k=k,s=s)[:,:-int(beta)]
        s2 = x2[:,:-int(beta)]
    else:
        s1 = shift_trajectory(x1,beta,k=k,s=s)[:,-int(beta):]
        s2 = x2[:,-int(beta):]

    # Show shifted 2D trajectory
    if show:
        vis.show_trajectory_2D(x1[:,:],s1[:,:-200],text=False,
        title='Original (left) vs Shifted (right), Beta={}, k={} and s={}'.format(beta,k,s))

    # Normalize points
    # s1, T1 = ep.normalize_2d_points(s1)
    # s2, T2 = ep.normalize_2d_points(s2)

    # Randomly sample 9 point correspondences
    num_solver = 9
    idx = np.random.choice(s1.shape[1], num_solver, replace=False)
    data = np.append(np.append(s1[:,idx],s2[:,idx],axis=0),ds[:,idx],axis=0)

    w = compute_beta(data)
    M = compute_beta_fundamental(data,param)

    if show:
        print("\nGround truth of beta: \n{}\n".format(beta))
        print("Possible solutions of beta: \n{}\n".format(w))

    return M


def iter_sync(x1,x2,param,it_max=100,p_min=0,p_max=8,threshold=1,maxiter=500,loRansac=False):
    '''
    Iterative algorithm beyond the solver for Beta and F
    '''

    # Load parameter
    k = param['k']
    s = param['s']

    # Initialize
    num_all = x1.shape[1]
    beta, inliers = np.zeros(it_max),np.zeros(it_max)
    F = np.zeros((it_max,9))
    beta[0], inliers[0], j, skip, it = 0,0,0,0,1
    p = p_min
    param['d'] = 2**p_min
    s1, s2 = x1, x2
    
    # Iteration
    while it<it_max:
        # no inprovement after trying all interpolation distance -> terminate
        if skip > p_max:
            return j, F[it-1], inliers[it-1]

        # Try d and -d
        R1 = compute_beta_fundamental_Ransac(s1,s2,param,threshold=threshold,maxiter=maxiter,loRansac=loRansac)
        param['d'] = -param['d']
        R2 = compute_beta_fundamental_Ransac(s1,s2,param,threshold=threshold,maxiter=maxiter,loRansac=loRansac)
        param['d'] = -param['d']

        # Compute both ratio of inliers
        beta_temp_1 = R1['model'][-1]
        beta_temp_2 = R2['model'][-1]
        
        if beta_temp_1 > 0:
            inliers_1 = sum(R1['inliers'] < s1.shape[1]-beta_temp_1-1) / (s1.shape[1]-abs(beta_temp_1))
        else:
            inliers_1 = sum(R1['inliers'] > -beta_temp_1) / (s1.shape[1]-abs(beta_temp_1))

        if beta_temp_2 > 0:
            inliers_2 = sum(R1['inliers'] < s1.shape[1]-beta_temp_2-1) / (s1.shape[1]-abs(beta_temp_2))
        else:
            inliers_2 = sum(R1['inliers'] > -beta_temp_2) / (s1.shape[1]-abs(beta_temp_2))

        # Choose the better result
        if max(inliers_1,inliers_2) < 0:
            inliers[it] = 0
        elif inliers_1 > inliers_2:
            inliers[it] = inliers_1
            beta[it] = beta_temp_1
            F[it] = R1['model'][:-1]
        else:
            inliers[it] = inliers_2
            beta[it] = beta_temp_2
            F[it] = R2['model'][:-1]

        # # Choose the better result
        # if R1['inliers'].shape[0] < R2['inliers'].shape[0]:
        #     R1 = R2

        # # Compute inliers, Beta, F
        # beta_temp = R1['model'][-1]
        # if abs(beta_temp) > s1.shape[1]:
        #     inliers[it] = 0
        # elif beta_temp > 0:
        #     inliers[it] = sum(R1['inliers'] < s1.shape[1]-beta_temp-1) / (s1.shape[1]-abs(beta_temp))
        #     beta[it] = beta_temp
        #     F[it] = R1['model'][:-1]
        # else:
        #     inliers[it] = sum(R1['inliers'] > -beta_temp) / (s1.shape[1]-abs(beta_temp))
        #     beta[it] = beta_temp
        #     F[it] = R1['model'][:-1]
        
        # ratio of inliers reaches a high threshold -> terminate
        if inliers[it] > 0.99:
            j = j + beta[it]
            print('Iteration of {} is finished, beta:{}, ratio of inliers:{:.3f}'.format(it, j, inliers[it]))
            return j, F[it], inliers[it]
        # Less inlier -> change interpolation distance d
        elif inliers[it] < inliers[it-1] or abs(j+beta[it])>num_all:
            if p < p_max:
                p += 1
            else:
                p = 0
            param['d'] = 2**p
            skip += 1
            print("skip:{}".format(skip))
        # More inlier -> shift trajectory
        else:
            j = j + beta[it]
            s2 = shift_trajectory(x2,j,k=k,s=s)
            if j >= 1:
                s1 = x1[:,:-int(j)]
                s2 = s2[:,:-int(j)]
            else:
                s1 = x1[:,-int(j):]
                s2 = s2[:,-int(j):]
            skip = 0
            print('Iteration of {} is finished, beta:{}, ratio of inliers:{:.3f}'.format(it, j, inliers[it]))
            it += 1
        
        print('d={}'.format(param['d']))


def search_sync(x1,x2,param,d_min=-10,d_max=10,threshold1=10,threshold2=1,maxiter=300,loRansac=False):

    # Brute-force running for different interpolation d
    inliers = 0
    d_all = np.arange(d_min,d_max+1)*20+1
    beta = None
    while True:
        for d in d_all:
            param['d'] = d
            result = compute_beta_fundamental_Ransac(x1,x2,param,threshold=threshold1,maxiter=maxiter,loRansac=loRansac)
            beta_temp = result['model'][-1]
            if abs(beta_temp) > x1.shape[1]:
                continue
            elif beta_temp > 0:
                inliers_temp = sum(result['inliers'] < x1.shape[1]-beta_temp-1) / (x1.shape[1]-abs(beta_temp))
            else:
                inliers_temp = sum(result['inliers'] > -beta_temp) / (x1.shape[1]-abs(beta_temp))

            # if abs(d) - abs(beta_temp) < 20 and abs(d) - abs(beta_temp) > 0:
            if abs(beta_temp-d) < 20:
                if inliers_temp > 0.5:
                    beta = beta_temp
                    inliers = inliers_temp
                    d_1 = d
                    break
                elif inliers_temp > inliers:
                    beta = beta_temp
                    inliers = inliers_temp
                    d_1 = d
                    print('The current estimated Beta is {}, with ratio of inliers of {}, d={}'.format(beta,inliers,d_1))
        if beta is not None:
            print('\nThe first stage: Beta is {}, with ratio of inliers of {}, d={}\n'.format(beta,inliers,d_1))
            break

    # Fine search using d=1 or d=-1, until no improvement in 3 iter or inlier > 0.99
    t, inliers = 0, 0
    while t<3 and inliers<0.999:
        s2 = shift_trajectory(x2,beta,k=param['k'],s=param['s'])
        if beta >= 1:
            s1 = x1[:,:-int(beta)]
            s2 = s2[:,:-int(beta)]
        else:
            s1 = x1[:,-int(beta):]
            s2 = s2[:,-int(beta):]

        param['d'] = 1
        result1 = compute_beta_fundamental_Ransac(s1,s2,param,threshold=threshold2,maxiter=maxiter,loRansac=loRansac)
        param['d'] = -1
        result2 = compute_beta_fundamental_Ransac(s1,s2,param,threshold=threshold2,maxiter=maxiter,loRansac=loRansac)

        # d = 1
        beta_temp_1 = beta + result1['model'][-1]
        if beta_temp_1 > 0:
            inliers_temp_1 = sum(result1['inliers'] < s1.shape[1]-beta_temp_1-1) / (s1.shape[1]-abs(beta_temp_1))
        else:
            inliers_temp_1 = sum(result1['inliers'] > -beta_temp_1) / (s1.shape[1]-abs(beta_temp_1))

        # d = -1
        beta_temp_2 = beta + result2['model'][-1]
        if beta_temp_2 > 0:
            inliers_temp_2 = sum(result2['inliers'] < s1.shape[1]-beta_temp_2-1) / (s1.shape[1]-abs(beta_temp_2))
        else:
            inliers_temp_2 = sum(result2['inliers'] > -beta_temp_2) / (s1.shape[1]-abs(beta_temp_2))
        
        # choose the better one
        if inliers_temp_1 > inliers_temp_2:
            inliers_temp = inliers_temp_1
            beta_temp = beta_temp_1
            F_temp = result1['model'][:-1]
            d_temp = 1
        else:
            inliers_temp = inliers_temp_2
            beta_temp = beta_temp_2
            F_temp = result2['model'][:-1]
            d_temp = -1

        # Compare to current max
        if inliers_temp > inliers:
            beta = beta_temp
            inliers = inliers_temp
            F = F_temp
            d = d_temp
            t = 0
            print('The current estimated Beta is {}, with ratio of inliers of {}, d={}'.format(beta,inliers,d))
        else:
            t+=1
    print('\nThe second stage: Beta is {}, with ratio of inliers of {}, d={}\n'.format(beta,inliers,d))
    return beta, F, inliers


if __name__ == "__main__":

    '''Load data'''

    # Load trajectory data
    X = np.loadtxt('./data/Real_Trajectory.txt')
    X = np.insert(X,3,1,axis=0)
    num_points = X.shape[1]

    # Smooth trajectory
    # t = np.arange(num_points, dtype=float)
    # spline = [UnivariateSpline(t, X[0], s=1),UnivariateSpline(t, X[1], s=1),UnivariateSpline(t, X[2], s=1)]
    # X[0], X[1], X[2] = spline[0](t), spline[1](t), spline[2](t)

    # Load camera parameter
    with open('./data/Synthetic_Camera.pickle', 'rb') as file:
        Camera =pickle.load(file)

    # Camara 1
    P1 = np.dot(Camera["K1"],np.hstack((Camera["R1"],Camera["t1"])))
    x1 = np.dot(P1,X)
    x1 /= x1[-1]
    img1 = Camera["img1"]

    # Camara 2
    P2 = np.dot(Camera["K2"],np.hstack((Camera["R2"],Camera["t2"])))
    x2 = np.dot(P2,X)
    x2 /= x2[-1]
    img2 = Camera["img2"]

    # show 3D trajectory
    # vis.show_trajectory_3D(X)


    '''Verify beta'''
    ################ 1. Single verification ################
    # verify_beta(x1,x2,0.5,{'d':1, 'k':1, 's':0})


    ################ 2 Algorithm ################
    # start=datetime.now()

    # beta = 193.07
    # param = {'k':1, 's':0}
    # s1 = shift_trajectory(x1,beta,k=param['k'],s=param['s'])
    # if beta >= 1:
    #     s1 = s1[:,:-int(beta)]
    #     s2 = x2[:,:-int(beta)]
    # else:
    #     s1 = s1[:,-int(beta):]
    #     s2 = x2[:,-int(beta):]
    
    # # Brute-force
    # # result = search_sync(s1,s2,param,threshold=10,maxiter=200,loRansac=True)
    # # Iterative 
    # result = iter_sync(s1,s2,param,threshold=10,maxiter=300,loRansac=True)
    # print('\nTime: ',datetime.now()-start)


    ################ 3. Solver for Beta and F, set d, test different Beta ################
    start=datetime.now()

    # Set parameters for initial triangulation through external parsing
    parser = argparse.ArgumentParser(description="Performance of solver")
    parser.add_argument('-d',help='Choose interpolation distance',default=1,type=int)
    parser.add_argument('-i',help='Iteration for a single d',default=10,type=int)
    parser.add_argument('-s',help='step of d',default=3,type=int)
    args = vars(parser.parse_args())

    # parameters
    beta_min = -54
    beta_max = 55
    beta_step = args['s']
    maxiter = 300
    threshold = 5
    LoRansac = False
    param = {'d':args['d'], 'k':1, 's':0}

    # Set number of iterations and whether to show individual results in each run
    it_max = args['i']
    show_single = False

    # Start computing...
    beta = np.arange(beta_min,beta_max,beta_step)
    results_all = np.zeros((3*it_max, len(beta)))

    for it in range(it_max):
        print('\n\n---------------Iteration {}---------------'.format(it+1))

        # Compute beta and F using Ransac
        print('Start iteraing...\n')
        results = {"beta":[],"beta_error":[],"ratio_inliers":[]}
        for i in range(len(beta)):
            s1 = shift_trajectory(x1,beta[i],k=param['k'],s=param['s'])
            if beta[i] >= 1:
                s1 = s1[:,:-int(beta[i])]
                s2 = x2[:,:-int(beta[i])]
            else:
                s1 = s1[:,-int(beta[i]):]
                s2 = x2[:,-int(beta[i]):]

            F, mask = cv2.findFundamentalMat(s1[:2].T,s2[:2].T,method=cv2.RANSAC,ransacReprojThreshold=5)
            mask = mask.reshape(-1,)
            estimate = compute_beta_fundamental_Ransac(s1,s2,param,threshold=threshold,maxiter=maxiter,loRansac=LoRansac)

            beta_temp = estimate['model'][-1]
            num_all = s1.shape[1]
            results['beta'].append(beta_temp)
            results['beta_error'].append(beta[i]-beta_temp)
            if beta_temp > 0:
                num_inliers = sum(estimate['inliers'] < num_all-beta_temp-1)
            else:
                num_inliers = sum(estimate['inliers'] > -beta_temp)
            results['ratio_inliers'].append(num_inliers / (num_all-abs(beta_temp)))

            if num_inliers / (num_all-abs(beta_temp)) > 1:
                print(num_inliers, num_all, beta_temp)

            print('Beta={:.2f} is finished, estimated Beta={:.2f}, ratio of inliers={:.3f}'.format(beta[i], results['beta'][-1], results['ratio_inliers'][-1]))

        # plot results
        if show_single:
            fig, ax = plt.subplots(1,3,sharex=True)
            fig.add_subplot(111, frameon=False)

            ax[0].plot(beta,results['beta'])
            ax[1].plot(beta,results['beta_error'])
            ax[2].plot(beta,results['ratio_inliers'])
            ax[0].set_title('Estimation of Beta')
            ax[1].set_title('Error of estimated Beta')
            ax[2].set_title('Ratio of inliers for each Beta')
            ax[0].set_ylabel('Estimated Beta')
            ax[1].set_ylabel('Error')
            ax[2].set_ylabel('Ratio of inliers')

            plt.xticks(np.linspace(beta_min,beta_max,11))
            plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            plt.grid(False)
            plt.xlabel("Ground Truth of Beta")
            fig.suptitle('Range of beta: {} to {} in step {}, Threshold: {}, MaxIter: {}, Using LoRansac: {}, k={}, s={}, d={}'.format(beta_min,
            beta_max,beta_step,threshold,maxiter,LoRansac,param['k'],param['s'],param['d']))
            plt.show()
        
        # Save intermediate results
        results_all[it*3:(it+1)*3] = np.array([results['beta'], results['beta_error'], results['ratio_inliers']])

        print('\nTime: ',datetime.now()-start)
    
    # Save final results in a pickle file
    interp_dist = {'beta_estimate': results_all[np.arange(it_max)*3],
                   'beta_error':    results_all[np.arange(it_max)*3+1],
                   'ratio_inliers': results_all[np.arange(it_max)*3+2],
                   'maxiter':       maxiter,
                   'threshold':     threshold,
                   'LoRansac':      LoRansac,
                   'd':             param['d']}

    with open('./data/solver/interpolation_distance_d'+str(args['d'])+'_cv.pkl','wb') as f:
        pickle.dump(interp_dist, f)

    # Plot results
    beta_estimate = np.mean(results_all[np.arange(it_max)*3], axis=0)
    beta_error =    np.mean(results_all[np.arange(it_max)*3+1], axis=0)
    ratio_inliers = np.mean(results_all[np.arange(it_max)*3+2], axis=0)

    # fig, ax = plt.subplots(1,3,sharex=True)
    # fig.add_subplot(111, frameon=False)

    # ax[0].plot(beta,beta_estimate)
    # ax[1].plot(beta,beta_error)
    # ax[2].plot(beta,ratio_inliers)
    # ax[0].set_title('Estimation of Beta')
    # ax[1].set_title('Error of estimated Beta')
    # ax[2].set_title('Ratio of inliers for each Beta')
    # ax[0].set_ylabel('Estimated Beta')
    # ax[1].set_ylabel('Error')
    # ax[2].set_ylabel('Ratio of inliers')

    # plt.xticks(np.linspace(beta_min,beta_max,11))
    # plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    # plt.grid(False)
    # plt.xlabel("Ground Truth of Beta")
    # fig.suptitle('Range of beta: {} to {} in step {}, Threshold: {}, MaxIter: {}, Using LoRansac: {}, d: {}, Iteration: {}'.format(beta_min,
    # beta_max,beta_step,threshold,maxiter,LoRansac,param['d'],it_max))
    # plt.show()

    print("Finish!\n")