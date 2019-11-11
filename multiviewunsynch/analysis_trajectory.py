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

data_file = ''
with open(data_file, 'rb') as file:
    flight = pickle.load(file)


print('Finish!')