import numpy as np
import cv2
# from matplotlib import pyplot as plt
import epipolar as ep

'''
This script tests the case of using 2D trajectory of drone to estimate F
'''

p1 = np.loadtxt('data/video_1_outputss.txt',skiprows=1,dtype=np.int32)
p2 = np.loadtxt('data/video_2_outputss.txt',skiprows=1,dtype=np.int32)

p = np.hstack((p1[153:153+1500,1:],p2[70:70+1500,1:]))
p = np.unique(p,axis=0)



print('finish')