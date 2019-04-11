import pickle
import numpy as np
import scipy.linalg
import epipolar as ep
import video
import ransac
from datetime import datetime
import visualization as vis
from matplotlib import pyplot as plt


def shift_trajectory(x,beta):
    '''
    Shift trajectory with given frames

    If beta is positiv, the last "int(beta)" frames will be dropped

    If beta is negative, the first "int(abs(beta))" frames will be dropped
    '''

    beta_dec = abs(beta)-int(abs(beta))
    beta_int = int(abs(beta))

    # tangent vector v
    ds = x[:,1:] - x[:,:-1]

    # Compute shifted data
    if beta > 0:
        if beta_dec:
            y_int = x[:,beta_int:-1]
            y_dec = ds[:,beta_int:] * beta_dec
            y = y_int + y_dec
        else:
            y = x[:,beta_int:]
    elif beta == 0:
        y = x[:,:-1]
    else:
        if beta_dec:
            y_int = x[:,1:x.shape[1]-beta_int]
            y_dec = -ds[:,:ds.shape[1]-beta_int] * beta_dec
            y = y_int + y_dec
        else:
            y = x[:,:-beta_int]

    return y


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


def compute_beta_fundamental(data):
    '''
    This function reads data of 9 points and return a list of possible solution for Beta and F

    It can be called by the Ransac function as a function handle 
    '''

    w = compute_beta(data)
    w[np.iscomplex(w)] = np.inf
    beta = w.real[np.isfinite(w)]

    # decompose input data
    s1 = data[:3,:]
    s2 = data[3:6,:]
    ds = data[6:,:]

    M = np.zeros((len(beta),10))
    for i in range(len(beta)):
        x2 = s2 + beta[i]*ds
        F = ep.compute_fundamental(s1,x2)
        M[i] = np.append(np.ravel(F),np.array([beta[i]]))

    return M


def compute_beta_fundamental_Ransac(x1,x2,threshold=10e-4,maxiter=500,verbose=False,loRansac=False):
    '''
    This function calls the Ransac for computing Beta and F and returns the best model and inliers.
    '''

    ds = x2[:,1:] - x2[:,:-1]
    data = np.append(np.append(x1[:,:-1],x2[:,:-1],axis=0),ds,axis=0)

    if loRansac:
        return ransac.loRansacSimple(compute_beta_fundamental,error_beta_fundamental,data,9,threshold,maxiter,verbose=verbose)
    else:
        return ransac.vanillaRansac(compute_beta_fundamental,error_beta_fundamental,data,9,threshold,maxiter,verbose=verbose)


def error_beta_fundamental(M,data):
    '''
    This function computes the Sampson error given Beta and F

    The alternative releases the shifted correspondence such that this does not have to be on the linear extension
    '''

    # decompose model
    F = M[:9].reshape((3,3))
    beta = M[9]

    # decompose input data
    x1 = data[:3,:]
    x2 = data[3:6,:] + beta*data[6:,:]

    # Alternative:
    if abs(beta) < data.shape[1]:
        x2_shift = shift_trajectory(data[3:6,:],beta)
        if beta >= 0:
            x2[:,:x2_shift.shape[1]] = x2_shift
        else:
            x2[:,data.shape[1]-x2_shift.shape[1]:] = x2_shift

    return ep.Sampson_error(x1,x2,F)


def verify_beta(x1,x2,beta,show=True):
    '''
    This is a simple function to show the functionality of computing Beta and F
    '''

    ds = x2[:,1:] - x2[:,:-1]
    s1 = shift_trajectory(x1,beta)
    num_data = s1.shape[1]
    s2 = x2[:,:num_data]
    ds = ds[:,:num_data]

    # Normalize points
    # s1, T1 = ep.normalize_2d_points(s1)
    # s2, T2 = ep.normalize_2d_points(s2)

    # Randomly sample 9 point correspondences
    num_solver = 9
    idx = np.random.choice(s1.shape[1], num_solver, replace=False)
    data = np.append(np.append(s1[:,idx],s2[:,idx],axis=0),ds[:,idx],axis=0)

    w = compute_beta(data)
    M = compute_beta_fundamental(data)

    if show:
        print("\nGround truth of beta: \n{}\n".format(beta))
        print("Possible solutions of beta: \n{}\n".format(w))

    return M


def verify_beta_fundamental_Ransac(x1,x2,beta,threshold=10e-4,maxiter=500,verbose=False,loRansac=False):
    '''
    This function verifies the computation of Beta and F (Beta should be given)
    '''

    ds = x2[:,1:] - x2[:,:-1]
    if beta >= 0:
        s1 = shift_trajectory(x1,beta)
        num_data = s1.shape[1]
        data = np.append(np.append(s1,x2[:,:num_data],axis=0),ds[:,:num_data],axis=0)
    else:
        s1 = shift_trajectory(x1,beta)
        s2 = shift_trajectory(x2,-np.floor(beta))
        data = np.append(np.append(s1[:,:-1],s2[:,:-1],axis=0),ds[:,int(-np.floor(beta)):],axis=0)

    if loRansac:
        return ransac.loRansacSimple(compute_beta_fundamental,error_beta_fundamental,data,9,threshold,maxiter,verbose=verbose)
    else:
        return ransac.vanillaRansac(compute_beta_fundamental,error_beta_fundamental,data,9,threshold,maxiter,verbose=verbose)


if __name__ == "__main__":

    '''Load data'''

    # Load trajectory data
    X = np.loadtxt('data/Real_Trajectory.txt')
    X_homo = np.insert(X,3,1,axis=0)

    # Load camera parameter
    with open('data/Synthetic_Camera.pickle', 'rb') as file:
        Camera =pickle.load(file)

    # Camara 1
    P1 = np.dot(Camera["K1"],np.hstack((Camera["R1"],Camera["t1"])))
    x1 = np.dot(P1,X_homo)
    x1 /= x1[-1]
    img1 = Camera["img1"]

    # Camara 2
    P2 = np.dot(Camera["K2"],np.hstack((Camera["R2"],Camera["t2"])))
    x2 = np.dot(P2,X_homo)
    x2 /= x2[-1]
    img2 = Camera["img2"]

    # show 3D trajectory
    vis.show_trajectory_3D(X)


    '''Verify beta'''
    # Single verification
    verify_beta(x1,x2,0.6)

    start=datetime.now()

    # parameters
    beta_min = -3
    beta_max = 0
    beta_step = 0.1
    maxiter = 500
    threshold = 0.01
    LoRansac = True

    # Compute beta and F using Ransac
    print('Start iteraing...\n')
    beta = np.arange(beta_min,beta_max,beta_step)
    results = {"beta":[],"beta_error":[],"numinliers":[]}
    for i in range(len(beta)):
        estimate = verify_beta_fundamental_Ransac(x1,x2,beta[i],threshold=threshold,maxiter=maxiter,loRansac=LoRansac)

        results['beta'].append(estimate['model'][-1])
        results['beta_error'].append(abs(estimate['model'][-1]-beta[i]))
        results['numinliers'].append(estimate['inliers'].shape[0])

        print('Beta={} is finished, estimated Beta={}, number of inliers={}'.format(beta[i],estimate['model'][-1],estimate['inliers'].shape[0]))

    # plot results
    fig, ax = plt.subplots(1,3,sharex=True)
    fig.add_subplot(111, frameon=False)

    ax[0].plot(beta,results['beta'])
    ax[1].plot(beta,results['beta_error'])
    ax[2].plot(beta,results['numinliers'])
    ax[0].set_title('Estimation of Beta')
    ax[1].set_title('Error of estimated Beta')
    ax[2].set_title('Number of inliers for each Beta')
    ax[0].set_ylabel('Estimated Beta')
    ax[1].set_ylabel('Error')
    ax[2].set_ylabel('Number of inliers')

    plt.xticks(np.linspace(beta_min,beta_max,11))
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Ground Truth of Beta")
    fig.suptitle('Range of beta: {} to {} in step {}, Threshold: {}, MaxIter: {}. Using LoRansac: {}'.format(beta_min,
    beta_max,beta_step,threshold,maxiter,LoRansac))
    plt.show()

    print('\nTime: ',datetime.now()-start)
    print("Finish!\n")