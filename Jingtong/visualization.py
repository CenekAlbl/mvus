import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def drawlines(img1,img2,lines,pts1,pts2):
    ''' 
    Function:
            draw lines and circles on image pair, specifically for epipolar lines
    Input:
            img1 = image on which we draw (epipolar) lines for the points in img2
            img2 = corresponding image
            lines = corresponding epipolar lines 
            pts1,pts2 = corresponding feature points
    Output:
            img1 = image1 with lines and circles
            img2 = image2 with circles 
    '''
    r,c = img1.shape

    # Convert grayscale to BGR if needed
    if len(img1.shape)==2:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    if len(img2.shape)==2:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    # Draw epipolar lines in image1 and corresponding feature points in both images
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def plotEpiline(img1, img2, pts1, pts2, F):
    '''
    Function:
            plot epipolar lines in both images, in form of N*2, type should be int
    Input:
            img1, img2 = image pair
            pts1, pts2 = corresponding feature points
            F = fundamental matrix
    Output:
            plot epipolar lines on image pair
    '''
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img3 = drawlines(img1,img2,lines1,pts1,pts2)[0]

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img5 = drawlines(img2,img1,lines2,pts2,pts1)[0]

    # Show results
    cv2.namedWindow('Epipolar lines in img1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Epipolar lines in img1',500,300)
    cv2.imshow('Epipolar lines in img1',img3)
    cv2.namedWindow('Epipolar lines in img2',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Epipolar lines in img2',500,300)
    cv2.imshow('Epipolar lines in img2',img5)
    cv2.waitKey(0)


def plot_epipolar_line(img1, img2, F, x1, x2):
    '''
    Plot epipolar lines of point correspondences.
    '''

    # pre-steps
    num = x1.shape[1]
    r1,c1 = img1.shape
    r2,c2 = img2.shape
    x1_coord = np.linspace(0,c1-1,100)
    x2_coord = np.linspace(0,c2-1,100)

    # Calculate epipolar lines
    line1 = np.dot(F,x1)
    line2 = np.dot(F.T,x2)

    # plot epipolar lines in img1, which are calculated using key points in img2
    plt.subplot(121),plt.imshow(img1,cmap='gray')
    for i in range(num):
        line = line2[:,i]
        y1_coord = np.array([(line[2]+line[0]*x)/(-line[1]) for x in x1_coord])
        idx = (y1_coord>=0) & (y1_coord<r1-1)
        plt.plot(x1_coord[idx],y1_coord[idx])
    plt.scatter(x1[0],x1[1])
    
    # plot epipolar lines in img2, which are calculated using key points in img1
    plt.subplot(122),plt.imshow(img2,cmap='gray')
    for i in range(num):
        line = line1[:,i]
        y2_coord = np.array([(line[2]+line[0]*x)/(-line[1]) for x in x2_coord])
        idx = (y2_coord>=0) & (y2_coord<r2-1)
        plt.plot(x2_coord[idx],y2_coord[idx])
    plt.scatter(x2[0],x2[1])

    plt.show()


def show_trajectory_2D(*x, color=False,line=False):

    num = len(x)
    for i in range(num):
        plt.subplot(1,num,i+1)
        plt.scatter(x[i][0],x[i][1],c=np.arange(x[i].shape[1])*color)
        if line:
            plt.plot(x[i][0],x[i][1])
        plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Y')
    plt.show()


def show_trajectory_3D(X,color=False,line=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter3D(X[0],X[1],X[2],c=np.arange(X.shape[1])*color)
    if line:
        ax.plot(X[0],X[1],X[2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == "__main__":
    # Load trajectory data
    X = np.loadtxt('data/Synthetic_Trajectory_generated.txt')
    X_homo = np.insert(X,3,1,axis=0)

    show_trajectory_3D(X,color=True)