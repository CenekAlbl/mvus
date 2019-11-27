import numpy as np
import pickle
import tools.visualization as vis
from datetime import datetime


data_file = ''
with open(data_file, 'rb') as file:
    flight = pickle.load(file)
