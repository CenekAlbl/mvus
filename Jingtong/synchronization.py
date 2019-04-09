import pickle
import numpy as np
import scipy.linalg
import epipolar as ep
import video
import visualization as vis
from matplotlib import pyplot as plt


def verify_beta_perfect(x1,x2,beta,show=False):
    '''
    This is the very first function to varify computing time shift beta

    based on generalized eigenvalue solution. The inpute point correspondences

    are manipulated in a way that a correct beta would fit all points perfectly


    Input:
            x1,x2: point correspondences (currently from synthetic data)
            beta: ground truth of beta

    Output:
            print out the possible solutions of beta
    '''

    # Define virtual correspondences
    num_all = x1.shape[1]
    s1 = x1

    # Perfect correspondences, meaningful only for beta > 0.5 according to visualization
    s2 = np.ones((3,num_all+1))
    s2[:2,0] = x2[:2,0] + (x2[:2,0]-x2[:2,1]) * 1
    for i in range(num_all):
        d = x2[:,i] - s2[:,i]
        s2[:,i+1] = s2[:,i] + 1/beta*d

    # # Approximate correspondences
    # s2 = np.ones((3,num_all))
    # s2[:2,0] = x2[:2,0] + (x2[:2,0]-x2[:2,1]) * 1
    # for i in range(num_all-1):
    #     d = x2[:,i+1] - x2[:,i]
    #     s2[:,i+1] = s2[:,i] + beta*d

    if show:
        # Show results of the adjusted trajectory
        vis.show_trajectory_2D(x2[:,:],s2[:,:],color=True)
        print('\nMaximal coordinate of the trajectory is {}'.format(s2.max()))

    # Normalize points
    s1, T1 = ep.normalize_2d_points(s1)
    s2, T2 = ep.normalize_2d_points(s2)

    # Compute approximation of tangent vector
    ds = s2[:,1:] - s2[:,:-1]

    # Randomly sample 9 point correspondences
    num_solver = 9
    idx = np.random.choice(num_all, num_solver, replace=False)
    s1 = s1[:,idx]
    s2 = s2[:,idx]
    ds = ds[:,idx]

    # Create design matrices A1, A2
    A1 = np.array([s1[0]*s2[0],s1[0]*s2[1],s1[0],
                    s1[1]*s2[0],s1[1]*s2[1],s1[1],
                    s2[0],s2[1],np.ones(num_solver)]).T

    A2 = np.array([s1[0]*ds[0],s1[0]*ds[1],np.zeros(num_solver),
                s1[1]*ds[0],s1[1]*ds[1],np.zeros(num_solver),
                ds[0],ds[1],np.zeros(num_solver)]).T

    # Compute eigenvalue
    w, vr = scipy.linalg.eig(A1,b=A2)

    if show:
        print("\nGround truth of beta: \n{}\n".format(beta))
        print("Possible solutions of beta: \n{}\n".format(-w))
    
    return -w


def verify_beta_real(x1,x2,beta,show=False):

    beta_dec = abs(beta)-int(abs(beta))
    beta_int = int(abs(beta))

    # Define virtual correspondences
    num_all = x1.shape[1]

    if beta_dec:
        s2 = np.ones((3,num_all+1))
        s2[:2,0] = x2[:2,0] + (x2[:2,0]-x2[:2,1]) * 1

        if beta >= 0:
            for i in range(num_all):
                d = x2[:,i] - s2[:,i]
                s2[:,i+1] = s2[:,i] + d/beta_dec
            s1 = shift_trajectory(x1,beta_int)
        else:
            raise ValueError("beta has to be positive")
    else:
        s1 = shift_trajectory(x1,beta_int)
        s2 = x2

    if show:
        # Show results of the adjusted trajectory
        vis.show_trajectory_2D(x2[:,:],s2[:,:],color=True)
        print('\nMaximal coordinate of the trajectory is {}'.format(s2.max()))

    # Normalize points
    s1, T1 = ep.normalize_2d_points(s1)
    s2, T2 = ep.normalize_2d_points(s2)

    # Compute approximation of tangent vector
    ds = s2[:,1:] - s2[:,:-1]

    # Randomly sample 9 point correspondences
    num_solver = 9
    idx = np.random.choice(s1.shape[1], num_solver, replace=False)
    s1 = s1[:,idx]
    s2 = s2[:,idx]
    ds = ds[:,idx]

    # Create design matrices A1, A2
    A1 = np.array([s1[0]*s2[0],s1[0]*s2[1],s1[0],
                    s1[1]*s2[0],s1[1]*s2[1],s1[1],
                    s2[0],s2[1],np.ones(num_solver)]).T

    A2 = np.array([s1[0]*ds[0],s1[0]*ds[1],np.zeros(num_solver),
                s1[1]*ds[0],s1[1]*ds[1],np.zeros(num_solver),
                ds[0],ds[1],np.zeros(num_solver)]).T

    # Compute eigenvalue
    w, vr = scipy.linalg.eig(A1,b=A2)

    if show:
        print("\nGround truth of beta: \n{}\n".format(beta))
        print("Possible solutions of beta: \n{}\n".format(-w))

    return -w


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
    if beta >= 0:
        if beta_dec:
            y_int = x[:,beta_int:-1]
            y_dec = ds[:,beta_int:] * beta_dec
            y = y_int + y_dec
        else:
            y = x[:,beta_int:]
    else:
        if beta_dec:
            y_int = x[:,beta_int+1:]
            y_dec = -ds[:,:ds.shape[1]-beta_int] * beta_dec
            # y_dec = -ds[:,1:ds.shape[1]-beta_int+1] * beta_dec
            y = y_int + y_dec
        else:
            y = x[:,:-beta_int]

    return y


if __name__ == "__main__":
    # Load trajectory data
    X = np.loadtxt('data/Synthetic_Trajectory_generated.txt')
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

    # verify beta
    beta = 1.7
    w = verify_beta_real(x1,x2,beta,show=True)

    iteration = 500
    beta_all = np.empty([iteration])
    for i in range(len(beta_all)):
        w = verify_beta_real(x1,x2,beta)
        w[np.iscomplex(w)] = np.inf
        w = w.real
        idx = np.argmin(abs(w-beta))
        if np.isfinite(w[idx]):
            beta_all[i] = w[idx]
    beta_err = abs(beta_all-beta)

    print('Mean of error of Beta is {}\n'.format(beta_err.mean()))
    print('Median of error of Beta is {}\n'.format(np.median(beta_err)))

    fig = plt.figure()
    ax = plt.subplot(121)
    ax.set_title("Histogram of Beta")
    plt.hist(beta_all, bins=range(-10,10))
    plt.xticks(np.arange(-10, 11, 1))
    
    ax = plt.subplot(122)
    ax.set_title("Histogram of error of Beta")
    plt.hist(beta_err, bins=range(0,20))
    plt.xticks(np.arange(0, 20, 1))
    plt.show()

    print("Finish!\n")