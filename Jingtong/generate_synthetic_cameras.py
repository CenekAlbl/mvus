import os,math,pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
This script generates synthetic camera(s) for a given trajectory.

Please set the number of cameras that should be generated.

Important:

When an image from a generated camera is displayed, PRESS "s" to save it, otherwise it will be ignored
'''

# Set the number of camera
num_cam = 2

# Load trajectory data
X = np.loadtxt('data/Synthetic_Trajectory_generated.txt')
X_homo = np.insert(X,3,1,axis=0)

# Show the 3D trajectory
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot3D(X[0],X[1],X[2])
ax.scatter3D(X[0],X[1],X[2])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# define a function for rotation matrix
def rotation(x,y,z):
    x,y,z = x/180*math.pi, y/180*math.pi, z/180*math.pi

    Rx = np.array([[1,0,0],[0,math.cos(x),-math.sin(x)],[0,math.sin(x),math.cos(x)]])
    Ry = np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
    Rz = np.array([[math.cos(z),-math.sin(z),0],[math.sin(z),math.cos(z),0],[0,0,1]])

    return np.dot(np.dot(Rz,Ry),Rx)

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
    R = rotation(r[0],r[1],r[2])
    t = 20*np.random.random_sample((3,1))-10

    P = np.dot(K,np.hstack((R,t)))
    x = np.dot(P,X_homo)
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
if not os.path.isdir("data"):
    os.mkdir("data")

file = open('data/Synthetic_Camera.pickle', 'wb')
pickle.dump(Camera, file)
file.close()

