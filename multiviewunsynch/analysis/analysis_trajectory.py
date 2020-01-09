import numpy as np
import pickle
import tools.visualization as vis
from datetime import datetime
from analysis.compare_gt import align_gt
from reconstruction import synchronization as sync
from analysis.compare_gt import align_gt


# Load trajectories
data_file = ''
with open(data_file, 'rb') as file:
    flight = pickle.load(file)

# Analysis


print('Finish!')
