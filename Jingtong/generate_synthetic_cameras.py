import os,math,pickle
import cv2
import numpy as np
import util
import visualization as vis
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
This script generates synthetic camera(s) for a given trajectory.

Please set the number of cameras that should be generated.

Important:

When an image from a generated camera is displayed, PRESS "s" to save it, otherwise it will be ignored
'''

# Set the number of camera
num_cam = 4

# Select the trajectory data
filename = './data/Synthetic_Trajectory_generated.txt'

# Load trajectory data
X = np.loadtxt(filename)
X_homo = np.insert(X,3,1,axis=0)

# Show the 3D trajectory
vis.show_trajectory_3D(X,color=True,line=True)

''' 
Start generating cameras
'''
num_temp = 1
Camera = {}

print("\nStart to generate cameras...\n")
while num_temp <= num_cam:
    f = np.random.randint(200,2000)
    px = np.random.randint(500,1500)
    py = px - np.random.randint(100,400)
    r = np.random.randint(-180,180,size=3)

    K = np.array([[f,0,px],
               [0,f,py],
               [0,0,1]])
    R = util.rotation(r[0],r[1],r[2])
    t = 20*np.random.random_sample((3,1))-10

    P = np.dot(K,np.hstack((R,t)))
    x = np.dot(P,X_homo)
    if x[2].min() > 0:
        x /= x[-1]

        if max(x[0])<2*px and min(x[0])>0 and max(x[1])<2*py and min(x[1])>0 \
        and max(x[0])>2*px-200 and min(x[0])<200 and max(x[1])>2*py-200 and min(x[1])<200 :

            print("got it!\n")
            print("K=",K)
            print("R=",R)
            print("t=",t)
            print("X-coordinates of all points fall into the range {} and {}".format(min(x[0]),max(x[0])))
            print("Y-coordinates of all points fall into the range {} and {}".format(min(x[1]),max(x[1])))
            print("\nPress 's' if you want to save this one, otherwise press anykey to continue")

            img = np.zeros((py*2,px*2),dtype=np.uint8)
            x = np.int16(x[:2])

            for i in range(x.shape[1]):
                img[x[1,i],x[0,i]]= 255

            cv2.imshow("Synthetic Image",img)
            k = cv2.waitKey(0)
            if k == ord("s"):
                Camera["K{}".format(num_temp)] = K
                Camera["R{}".format(num_temp)] = R
                Camera["t{}".format(num_temp)] = t
                Camera["img{}".format(num_temp)] = img

                num_temp += 1
                cv2.destroyAllWindows()
            else:
                cv2.destroyAllWindows()

            if num_temp <= num_cam:
                print("\nGenerating the next camera...\n")
            else:
                print("\nFinished!\n")

# save the camera(s) in a pickle file
if not os.path.isdir("./data"):
    os.mkdir("./data")

with open('./data/Synthetic_Camera.pickle', 'wb') as f:
    pickle.dump(Camera, f)


