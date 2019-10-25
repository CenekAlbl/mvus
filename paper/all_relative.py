import pickle
import numpy as np
import visualization as vis

Traj = np.loadtxt('./data/icra/GT_position/GT_ENU.txt').T
vis.show_trajectory_3D(Traj,line=False,color=False)

print('Finish !!')