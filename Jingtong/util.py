import math
import numpy as np


def mapminmax(x,ymin,ymax):
    return (ymax-ymin)*(x-min(x))/(max(x)-min(x)) + ymin

def rotation(x,y,z):
    x,y,z = x/180*math.pi, y/180*math.pi, z/180*math.pi

    Rx = np.array([[1,0,0],[0,math.cos(x),-math.sin(x)],[0,math.sin(x),math.cos(x)]])
    Ry = np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
    Rz = np.array([[math.cos(z),-math.sin(z),0],[math.sin(z),math.cos(z),0],[0,0,1]])

    return np.dot(np.dot(Rz,Ry),Rx)