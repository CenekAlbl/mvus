import numpy as np
import pickle
import epipolar as ep
import util
import cv2
import skimage
import common
import visualization as vis

# Load trajectory data
X = np.loadtxt('./data/Synthetic_Trajectory_generated.txt')
X = np.insert(X,3,1,axis=0)
num_points = X.shape[1]

# vis.show_trajectory_3D(X)

# Load camera parameter
with open('./data/Synthetic_Camera.pickle', 'rb') as file:
    Camera = pickle.load(file)

# 2D trajectories
num_cam = int(len(Camera)/4)
x = []
for i in range(num_cam):
    P_i = eval('np.dot(Camera["K{}"],np.hstack((Camera["R{}"],Camera["t{}"])))'.format(i+1,i+1,i+1))
    x_i = np.dot(P_i,X)
    x_i /= x_i[-1]
    x.append(x_i)

# Save ground truth of R and t
for i in range(num_cam):
    Rt_i = eval('np.hstack((Camera["R{}"],Camera["t{}"]))'.format(i+1,i+1))
    np.savetxt('./data/Rt_{}.txt'.format(i+1),Rt_i,fmt='%.4f')

# Pairwise camera poses
E = {}
for i in range(num_cam-1):
    K_i = eval('Camera["K{}"]'.format(i+1))

    for j in range(i+1,num_cam):
        K_j = eval('Camera["K{}"]'.format(j+1))

        F,mask = ep.computeFundamentalMat(x[i],x[j])
        E_ij = np.linalg.multi_dot([K_j.T, F, K_i])
        
        X_ij, Rt = ep.triangulate_from_E(E_ij,K_i,K_j,x[i],x[j])
        np.savetxt('./data/Rt_{}{}.txt'.format(i+1,j+1),Rt,fmt='%.4f')

        exec('E["E_{}{}"]=E_ij'.format(i+1,j+1))

# Show triangulation
vi,vj = 2,3
P_i = common.Camera(P=eval('np.dot(Camera["K{}"],np.hstack((Camera["R{}"],Camera["t{}"])))'.format(vi,vi,vi)))
P_j = common.Camera(P=eval('np.dot(Camera["K{}"],np.hstack((Camera["R{}"],Camera["t{}"])))'.format(vj,vj,vj)))
P_i.center()
P_j.center()

X_ij, Rt = eval("ep.triangulate_from_E(E['E_{}{}'],Camera['K{}'],Camera['K{}'],x[{}],x[{}])".format(vi,vj,vi,vj,vi-1,vj-1))
R_diff = np.dot(Rt[:,:3],P_i.R) - P_j.R
print(R_diff)

P1 = common.Camera(P=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
P2 = common.Camera(P=Rt)
P1.center()
P2.center()

T = eval('np.vstack((np.hstack((Camera["R{}"],Camera["t{}"])), np.array([0,0,0,1])))'.format(vi,vi))
P_i = eval('np.dot(Camera["K{}"],np.hstack((Camera["R{}"],Camera["t{}"])))'.format(vi,vi,vi))
# P_j = eval('np.dot(Camera["K{}"],np.hstack((np.dot(Rt[:,:3],Camera["R{}"]),Rt[:,3].reshape((-1,1))+Camera["t{}"])))'.format(vj,vi,vi)) 
P_j = eval('np.dot(Camera["K{}"],np.dot(np.vstack((Rt,np.array([0,0,0,1]))),T)[:3])'.format(vj)) 
X_ij = ep.triangulate(x[vi-1],x[vj-1],P_i,P_j)

Rt[:,3] += np.array([0.1, 0.1, 0.1]) 
P_k = eval('np.dot(Camera["K{}"],np.dot(np.vstack((Rt,np.array([0,0,0,1]))),T)[:3])'.format(vj))
X_ik = ep.triangulate(x[vi-1],x[vj-1],P_i,P_k)

# vis.show_trajectory_3D(X,X_ij,X_ik)

transform =  util.umeyama(X_ij[:3,:17].T,X[:3,:17].T,estimate_scale=True)
X_ij_t = np.dot(transform,X_ij)

# vis.show_trajectory_3D(X,X_ij,X_ij_t)

print('Finished')