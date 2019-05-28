import numpy as np
import cv2
import epipolar as ep
import visualization as vis
import util
import pickle
from datetime import datetime
import synchronization
from scipy.interpolate import UnivariateSpline
import scipy.io as scio
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load data
traj_1 = np.loadtxt('./data/fixposition/c3_f2.txt',skiprows=1,dtype=np.int32)
traj_2 = np.loadtxt('./data/fixposition/c4_f2.txt',skiprows=1,dtype=np.int32)

# Truncate data
start, end = 0, 1300
x1 = util.homogeneous(traj_1[start:end,:].T)
x2 = util.homogeneous(traj_2[start:end,:].T)

# Compute beta
start=datetime.now()

param = {'k':1, 's':0}
beta, F, inliers = synchronization.search_sync(x1,x2,param,d_min=-20,d_max=20,threshold1=5,threshold2=2)
# beta, F, inliers = synchronization.iter_sync(x1,x2,param,threshold=5)

print('\nTime: ',datetime.now()-start)



print('Finished !')