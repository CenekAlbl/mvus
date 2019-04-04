import pickle
import numpy as np
import scipy.linalg
import epipolar as ep
import video


def verify_beta_perfect(x1,x2,beta):
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
    s2 = np.ones((3,num_all+1))

    s2[:2,0] = x2[:2,0] - np.array([beta*10,beta*10])
    for i in range(num_all):
        d = x2[:,i] - s2[:,i]
        s2[:,i+1] = s2[:,i] + 1/beta*d

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

    print("\nGround truth of beta: \n{}\n".format(beta))
    print("Possible solutions of beta: \n{}\n".format(-w))


def verify_beta_real(x1,x2,beta):

    # Normalize points
    x1, T1 = ep.normalize_2d_points(x1)
    x2, T2 = ep.normalize_2d_points(x2)

    # # Make sure beta is integer
    # beta = round(beta)

    # # Compute approximation of tangent vector
    # ds = x2[:,1:] - x2[:,:-1]

    # # Shift input data according to beta
    # if beta == 0:
    #     s1 = s1[:,:-1]
    #     s2 = s2[:,:-1]
    # elif beta == 1:
    #     s1 = s1[:,1:]
    #     s2 = s2[:,:-1]
    # else:
    #     s1 = s1[:,beta:]
    #     s2 = s2[:,:-beta]
    #     ds = ds[:,:-beta+1]

    # Shift both trajectory according to beta
    s1 = shift_trajectory(s1,beta)
    s2 = shift_trajectory(s2,beta)

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

    print("\nGround truth of beta: \n{}\n".format(beta))
    print("Possible solutions of beta: \n{}\n".format(-w))


def shift_trajectory(x,beta):
    assert beta>0, 'Currently only works for beta>0'

    # Split beta into integer and decimal part
    dec = beta - int(beta)
    beta = int(beta)

    # tangent vector v
    ds = x[:,1:] - x[:,:-1]

    # Compute shifted data
    y_int = x[:,beta:-1]
    y_dec = ds[:,beta:] * dec
    y = y_int + y_dec

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
    beta = 3
    verify_beta_real(x1,x2,beta)