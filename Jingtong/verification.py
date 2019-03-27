import pickle
import cv2
import numpy as np

# Load trajectory data
X = np.loadtxt('data/Synthetic_Trajectory.txt')
X_homo = np.insert(X.T,3,1,axis=0)

# Load camera parameter
with open('data/Synthetic_Camera.pickle', 'rb') as file:
    Camera =pickle.load(file)

# Camara 1
P1 = np.dot(Camera["K1"],np.hstack((Camera["R1"],Camera["t1"])))
x1 = np.dot(P1,X_homo)
x1 /= x1[-1]

# Camara 2
P2 = np.dot(Camera["K2"],np.hstack((Camera["R2"],Camera["t2"])))
x2 = np.dot(P2,X_homo)
x2 /= x2[-1]

# Use a certain percentage of data to compute F, the rest for validation
part = 0.7
num_all = x1.shape[1]
num_train = int(part*num_all)
num_val = num_all - num_train

idx = np.random.permutation(num_all)
x1_train = x1[:,idx[:num_train]]
x1_val = x1[:,idx[num_train:]]
x2_train = x2[:,idx[:num_train]]
x2_val = x2[:,idx[num_train:]]
print("\n{} points are used to compute F, {} points to validate by calculating x\'Fx".format(num_train,num_val))

# Compute fundamental matrix F
F,mask = cv2.findFundamentalMat(x1_train[:2].T, x2_train[:2].T, method=cv2.FM_RANSAC)
pts1 = x1_train[:2].T[mask.ravel()==1]
pts2 = x2_train[:2].T[mask.ravel()==1]

residual = np.diag(np.dot(np.dot(x2_val.T, F),x1_val))
print("\nThe maxmial residual is {}".format(residual.max()))

