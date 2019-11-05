# This script trys to solve the BA bug for the 6th camera

import numpy as np
import util
import epipolar as ep
import synchronization
# import common
import transformation
import scipy.io as scio
import pickle
import argparse
import copy
import cv2
import matplotlib.pyplot as plt
import visualization as vis
from datetime import datetime
from scipy.optimize import least_squares
from scipy import interpolate


with open('./data/paper/fixposition/trajectory/flight_spline_1.pkl', 'rb') as file:
    flight = pickle.load(file)

with open('./data/paper/fixposition/trajectory/flight_spline_2.pkl', 'rb') as file:
    flight_2 = pickle.load(file)


vis.show_trajectory_3D(flight.traj[1:], flight_2.traj[1:], line=False)


print('Finish!')