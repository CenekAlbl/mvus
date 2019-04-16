import pickle
import numpy as np
import scipy.linalg
import epipolar as ep
import video
import ransac
from datetime import datetime
import visualization as vis
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
        F = ep.compute_fundamental(s1,x2)
        M[i] = np.append(np.ravel(F),np.array([beta[i]*param['d']]))

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
        return ransac.loRansacSimple(compute_beta_fundamental,error_beta_fundamental,data,9,param,threshold,maxiter,verbose=verbose)
    else:
        return ransac.vanillaRansac(compute_beta_fundamental,error_beta_fundamental,data,9,threshold,maxiter,verbose=verbose)


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
    if d>0:
        s1 = shift_trajectory(x1,beta,k=k,s=s)[:,:-d]
        s2 = x2[:,:-d]
    else:
        s1 = shift_trajectory(x1,beta,k=k,s=s)[:,-d:]
        s2 = x2[:,-d:]

    # Show shifted 2D trajectory
    if show:
        vis.show_trajectory_2D(x1[:,:100],s1[:,:100],text=True,
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

    beta, inliers = np.zeros(it_max),np.zeros(it_max)
    F = np.zeros((it_max,9))
    beta[0], inliers[0], j, skip, it = 0,0,0,0,1
    p = p_min
    param['d'] = 2**p_min
    x2_shifted = x2
    
    while it<it_max:
        R1 = compute_beta_fundamental_Ransac(x1,x2_shifted,param,threshold=threshold,maxiter=maxiter,loRansac=loRansac)
        param['d'] = -param['d']
        R2 = compute_beta_fundamental_Ransac(x1,x2_shifted,param,threshold=threshold,maxiter=maxiter,loRansac=loRansac)
        param['d'] = -param['d']

        if R1['inliers'].shape[0] < R2['inliers'].shape[0]:
            R1 = R2
        inliers[it] = R1['inliers'].shape[0]
        beta[it] = R1['model'][-1]
        F[it] = R1['model'][:-1]

        if skip > p_max:
            return F[it-1], j+beta[it-1]
        elif inliers[it] < inliers[it-1]:
            if p < p_max:
                p += 1
            else:
                p = 0
            param['d'] = 2**p
            skip += 1
            print("skip:{}".format(skip))
        else:
            j = j + beta[it]
            x2_shifted = shift_trajectory(x2,j,k=k,s=s)
            skip = 0
            print('Iteration of {} is finished, beta:{}, ratio of inliers:{:.3f}'.format(it, j, inliers[it]/x1.shape[1]))
            it += 1


if __name__ == "__main__":

    '''Load data'''

    # Load trajectory data
    X = np.loadtxt('data/Real_Trajectory.txt')
    X = np.insert(X,3,1,axis=0)
    num_points = X.shape[1]

    # Smooth trajectory
    t = np.arange(num_points, dtype=float)
    spline = [UnivariateSpline(t, X[0], s=1),UnivariateSpline(t, X[1], s=1),UnivariateSpline(t, X[2], s=1)]
    X[0], X[1], X[2] = spline[0](t), spline[1](t), spline[2](t)

    # Load camera parameter
    with open('data/Synthetic_Camera.pickle', 'rb') as file:
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
    vis.show_trajectory_3D(X)


    '''Verify beta'''
    # 1. Single verification
    # verify_beta(x1,x2,5,{'d':-10, 'k':1, 's':0})


    # 2. Iterative algorithm
    start=datetime.now()

    beta = 100
    param = {'d':1, 'k':1, 's':0}
    s1 = shift_trajectory(x1,beta,k=param['k'],s=param['s'])
    result = iter_sync(s1,x2,param,maxiter=100,loRansac=True)

    print('\nTime: ',datetime.now()-start)


    # 3. Solver for Beta and F
    start=datetime.now()

    # parameters
    beta_min = -20
    beta_max = 20
    beta_step = 1
    maxiter = 500
    threshold = 0.1
    LoRansac = True
    param = {'d':-10, 'k':3, 's':0.1}

    # Compute beta and F using Ransac
    print('Start iteraing...\n')
    beta = np.arange(beta_min,beta_max,beta_step)
    results = {"beta":[],"beta_error":[],"ratio_inliers":[]}
    for i in range(len(beta)):
        s1 = shift_trajectory(x1,beta[i],k=param['k'],s=param['s'])
        s2 = x2
        estimate = compute_beta_fundamental_Ransac(s1,s2,param,threshold=threshold,maxiter=maxiter,loRansac=LoRansac)

        results['beta'].append(estimate['model'][-1])
        results['beta_error'].append(beta[i]-estimate['model'][-1])
        results['ratio_inliers'].append(estimate['inliers'].shape[0] / num_points)

        print('Beta={} is finished, estimated Beta={}, ratio of inliers={}'.format(beta[i], results['beta'][-1], results['ratio_inliers'][-1]))

    # plot results
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

    print('\nTime: ',datetime.now()-start)
    print("Finish!\n")