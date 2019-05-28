# Classes that are common to the entire project
import numpy as np
import util
import epipolar as ep
import cv2
import visualization as vis
import common

K = np.array([[1200,0,1000],[0,1200,600],[0,0,1]],dtype=np.float)
R = util.rotation(-20,50,150)
t = np.array([0.3,0.1,0.2])
cam = common.Camera(K=K,R=R,t=t)
cam.compose()

while True:
    a = util.homogeneous(np.random.randn(3,1000))
    b = np.dot(cam.P,a)
    b = b/b[-1]
    c = (b[0]<K[0,2]) & (b[0]>0) & (b[1]<K[1,2]) & (b[1]>0)

    if sum(c)>30:
        X = a[:,c]
        x = b[:,c]
        break

N = X.shape[1]
X[:3] = X[:3] + np.random.randn(3,N) * 0.01
print('Mean reprojection error: ',np.mean(ep.reprojection_error(x,cam.projectPoint(X))))

objectPoints = np.ascontiguousarray(X[:3].T).reshape((N,1,3))
imagePoints  = np.ascontiguousarray(x[:2].T).reshape((N,1,2))
retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints,imagePoints,cam.K,None,reprojectionError=10)
R_cv = cv2.Rodrigues(rvec)[0]
print(util.rotation_decompose(R_cv))

print('Finished')