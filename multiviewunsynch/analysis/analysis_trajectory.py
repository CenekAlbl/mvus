import numpy as np
import pickle
import tools.visualization as vis
from datetime import datetime


# Load trajectories
data_file_1 = './data/fixposition/trajectory/flight_rs_0.pkl'
with open(data_file_1, 'rb') as file:
    flight_1 = pickle.load(file)

data_file_2 = './data/fixposition/trajectory/flight_rs_1.pkl'
with open(data_file_2, 'rb') as file:
    flight_2 = pickle.load(file)

# Analysis
error_1 = np.sqrt(np.sum((flight_1.gps['gps']-flight_1.gps['traj'])**2,axis=0))
gps_idx_1 = flight_1.gps['gps_idx']

error_2 = np.sqrt(np.sum((flight_2.gps['gps']-flight_2.gps['traj'])**2,axis=0))
gps_idx_2 = flight_2.gps['gps_idx']

# vis.error_traj(flight.gps['traj'], error, thres=1, text=None)

print('Finish!')
