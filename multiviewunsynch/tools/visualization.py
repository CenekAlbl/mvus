# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import cv2
from tools import util
import pickle
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from thirdparty.camera_calibration_show_extrinsics import create_camera_model, transform_to_matplotlib_frame
from .util import homogeneous

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
    r,c = img1.shape[:2]

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
    img1 = img1.copy()
    img2 = img2.copy()
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
    # cv2.namedWindow('Epipolar lines in img1',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Epipolar lines in img1',500,300)
    # cv2.imshow('Epipolar lines in img1',img3)
    cv2.imwrite('epipolar_lines_img1.png', img3)
    # cv2.namedWindow('Epipolar lines in img2',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Epipolar lines in img2',500,300)
    # cv2.imshow('Epipolar lines in img2',img5)
    cv2.imwrite('epipolar_lines_img2.png', img5)

    # cv2.waitKey(0)


def plot_epipolar_line(img1, img2, F, x1, x2):
    '''
    Plot epipolar lines of point correspondences.
    '''
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # pre-steps
    num = x1.shape[1]
    r1,c1 = img1.shape[:2]
    r2,c2 = img2.shape[:2]
    x1_coord = np.linspace(0,c1-1,100)
    x2_coord = np.linspace(0,c2-1,100)

    # Calculate epipolar lines
    line1 = np.dot(F,x1)
    line2 = np.dot(F.T,x2)

    # plot epipolar lines in img1, which are calculated using key points in img2
    plt.subplot(121),plt.imshow(img1,cmap='gray')
    for i in range(x2.shape[1]):
        line = line2[:,i]
        y1_coord = np.array([(line[2]+line[0]*x)/(-line[1]) for x in x1_coord])
        idx = (y1_coord>=0) & (y1_coord<r1-1)
        plt.plot(x1_coord[idx],y1_coord[idx])
    plt.scatter(x1[0],x1[1])
    
    # plot epipolar lines in img2, which are calculated using key points in img1
    plt.subplot(122),plt.imshow(img2,cmap='gray')
    for i in range(x1.shape[1]):
        line = line1[:,i]
        y2_coord = np.array([(line[2]+line[0]*x)/(-line[1]) for x in x2_coord])
        idx = (y2_coord>=0) & (y2_coord<r2-1)
        plt.plot(x2_coord[idx],y2_coord[idx])
    plt.scatter(x2[0],x2[1])
    plt.savefig('epipolar_lines.png')
    plt.show()


def show_trajectory_2D(*x, title=None,color=True,line=False,text=False):
    plt.figure(figsize=(12, 10))
    num = len(x)
    for i in range(num):
        plt.subplot(1,num,i+1)
        plt.scatter(x[i][0],x[i][1],c=np.arange(x[i].shape[1])*color)
        if line:
            plt.plot(x[i][0],x[i][1])
        if text:
            for j in range(len(x[i][0])):
                plt.text(x[i][0,j], x[i][1,j], str(j), color='red',fontsize=12)

        # plt.gca().set_xlim([0,1920])
        # plt.gca().set_ylim([0,1080])

        plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Y')
    if title:
        plt.suptitle(title)
    plt.show()


def show_trajectory_3D(*X, title=None,color=True,line=False):
    fig = plt.figure(figsize=(12, 10))
    num = len(X)
    for i in range(num):
        ax = fig.add_subplot(1,num,i+1,projection='3d')
        
        if color:
            # ax.scatter3D(X[i][0],X[i][1],X[i][2],c='r')
            ax.scatter3D(X[i][0],X[i][1],X[i][2],c=np.arange(X[i].shape[1])*color)
        else:
            ax.scatter3D(X[i][0],X[i][1],X[i][2])

        if line:
            ax.plot(X[i][0],X[i][1],X[i][2])
        plt.xlabel('X')
        plt.ylabel('Y')
    if title:
        plt.suptitle(title)
    # plt.axis('off')
        plt.savefig(title+'.png')
    else:
        plt.savefig('reconstructed_trajectory.png')
    plt.show()


def show_2D_all(*x, title=None,color=True,line=True,text=False, bg=None, output_dir='', label=[], cam_center=[]):
    plt.figure(figsize=(12, 10))
    if bg is not None:
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        plt.imshow(bg)
        h,w,_ = bg.shape
        plt.xlim([0,w])
        plt.ylim([h,0])

    num = len(x)
    for i in range(num):
        # plt.subplot(1,num,i+1)

        c = ['r','b','orange','g']
        m = ['o','x','o','+']
        if len(label) == 0:
            label = ['Raw points', 'Reconstruction points','Raw detections', 'Reconstructed trajectories']
        if color:
            plt.scatter(x[i][0],x[i][1],c=c[i],marker=m[i],label=label[i])
        else:
            plt.scatter(x[i][0],x[i][1],c=c[i])
        # plt.scatter(x[i][0],x[i][1],c=np.arange(x[i].shape[1])*color)
        if line:
            plt.plot(x[i][0],x[i][1])
        if text:
            for j in range(len(x[i][0])):
                plt.text(x[i][0,j], x[i][1,j], str(j), color='red',fontsize=12)
        if len(cam_center) == 2:
            # plot camera position
            plt.scatter(cam_center[0], cam_center[1], c='c',marker='*')

        # plt.gca().set_xlim([0,1920])
        # plt.gca().set_ylim([0,1080])
        # plt.gca().invert_yaxis()

        plt.xlabel('X')
        plt.ylabel('Y')
    
    plt.legend(loc=1)

    if title:
        plt.suptitle(title)
        plt.savefig(output_dir+title+'.png')
    else:
        plt.savefig(output_dir+'reprojected.png')
    # plt.show()


def draw_camera_extrinsics(flight, ax, scale_focal=40):
    colors = [ cm.jet(x) for x in 100*np.random.rand(flight.numCam)]
    # loop through all the cameras
    for i, cam in enumerate(flight.cameras):
        # width and height of the camera
        cam_height, cam_width, _ = cam.img.shape
        # get the camera frame model
        X_cam_model = create_camera_model(cam.K, cam_width/2, cam_height/2, scale_focal, draw_frame_axis=True)
        cMo = np.eye(4)
        cMo[:3,:3] = cam.R
        cMo[:3,-1] = cam.t

        # print(len(X_cam_model))

        for k, X_cam_part in enumerate(X_cam_model):
            X = np.zeros_like(X_cam_part)
            for j in range(X_cam_part.shape[1]):
                X[0:4,j] = transform_to_matplotlib_frame(cMo, X_cam_part[0:4,j], True)
            if len(X_cam_model) == 8 and k == 5:
                ax.plot3D(X[0,:], X[1,:], X[2,:], color='r')
            elif len(X_cam_model) == 8 and k == 6:
                ax.plot3D(X[0,:], X[1,:], X[2,:], color='g')
            elif len(X_cam_model) == 8 and k == 7:
                ax.plot3D(X[0,:], X[1,:], X[2,:], color='b')
            else:
                ax.plot3D(X[0,:], X[1,:], X[2,:], color=colors[i])
        
        C_cam = np.dot(-cam.R.T, cam.t.reshape(-1,1)).ravel()

        ax.scatter3D(C_cam[0], C_cam[1], C_cam[2], c=colors[i])
        ax.text(C_cam[0], C_cam[1], C_cam[2], 'Camera '+str(i), color=colors[i])


def show_3D_all(*X, title=None,color=True,line=True,flight=None, output_dir='',label=[]):
    fig = plt.figure(figsize=(20, 15))
    num = len(X)
    ax = fig.add_subplot(111,projection='3d')

    # ax.set_zlim(-10,10)
    # ax.set_xlim(-10,10)
    # ax.set_ylim(-10,10)

    for i in range(num):
        if color:
            c = ['r','b','orange','g']
            m = ['o','x','o','+']
            if len(label) == 0:
                label = ['RTK ground truth', 'Reconstruction Spline']
            if i is 0:
                # ax.scatter3D(X[i][0],X[i][1],X[i][2],s=60,c=c[i],marker='o',label=label[i])
                ax.scatter3D(X[i][0],X[i][1],X[i][2],s=60,c=c[i%len(c)],marker=m[i%len(m)],label=label[i%len(label)])
            else:
                ax.scatter3D(X[i][0],X[i][1],X[i][2],s=60,c=c[i%len(c)],marker=m[i%len(m)],label=label[i%len(label)])
                # ax.plot(X[i][0],X[i][1],X[i][2],c=c[i])

        else:
            ax.scatter3D(X[i][0],X[i][1],X[i][2])

        if line:
            ax.plot(X[i][0],X[i][1],X[i][2])
        plt.xlabel('X')
        plt.ylabel('Y')
    
    # if the flight is provided, also draw the cameras and the reconstructed trajectories
    if flight is not None:
        draw_camera_extrinsics(flight, ax, scale_focal=.5)
        # if exists trajectory, also plot it
        if len(flight.traj) > 0:
            if color:
                ax.scatter3D(flight.traj[1], flight.traj[2], flight.traj[3], c=np.arange(flight.traj.shape[1])*color)
            else:
                ax.scatter3D(flight.traj[1], flight.traj[2], flight.traj[3])
            if line:
                ax.plot(flight.traj[1], flight.traj[2], flight.traj[3])

    if title:
        plt.suptitle(title)
    
    # ax.set_xlabel('East [m]',fontsize=20)
    # ax.set_ylabel('North [m]',fontsize=20)
    # ax.set_zlabel('Up [m]',fontsize=20)
    ax.set_xlabel('X',fontsize=20)
    ax.set_ylabel('Y',fontsize=20)
    ax.set_zlabel('Z',fontsize=20)

    ax.view_init(elev=30,azim=-50)
    lgnd = ax.legend(loc=8, prop={'size': 15})
    for handle in lgnd.legendHandles:
        handle.set_sizes([100])
    # plt.axis('off')
    if title:
        plt.savefig(output_dir+title+'reconstructed_scene')
    else:
        plt.savefig(output_dir+'reconstructed_scene.png')
    plt.show()


def show_spline(*spline, title=None):

    num = len(spline)
    for i in range(num):
        plt.subplot(1,num,i+1)
        x, y = spline[i][0], spline[i][1]
        plt.scatter(x[0],x[1],s=10,label='Origin')
        plt.plot(y[0],y[1],c='r',label='Spline interpolation')

        plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

    if title:
        plt.suptitle(title)
    plt.show()


def error_hist(error,bins=None,title=None,label=None):

    assert len(error.shape)==1, 'Input must be a 1D array'

    if not bins:    bins = np.arange(0,70,5)
    if not title:   title = 'Error histogram [cm]'
    if not label:   label = 'Spline'

    plt.figure(figsize=(20, 15))
    plt.hist(error.T*100, bins, histtype='bar', color='b',label=label)
    plt.xticks(bins,fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(loc=1, prop={'size': 30})
    plt.title(title, fontsize=25)
    plt.show()


def error_traj(traj,error,thres=0.5,title=None,colormap='Wistia',size=100, text=None):

    assert len(error.shape)==1, 'Input must be a 1D array'
    assert traj.shape[0]==3, 'Input must be a 3D array'
    assert len(error)==traj.shape[1], 'Error must have the same shape as the trajectory'

    if not title:   title = 'Error over the trajectory [cm]'

    fig = plt.figure(figsize=(20, 15))
    plt.set_cmap(colormap)
    ax = fig.add_subplot(111,projection='3d')
    sc = ax.scatter3D(traj[0],traj[1],traj[2],c=error*100,marker='o',s=size)

    # Plot the timestamp of points that have large errors
    if text is not None:
        assert len(text)==len(error), 'Wrong number of timestamps'
        text = text.astype(int)
        for i in range(len(text)):
            if error[i]>thres:
                ax.text(traj[0,i], traj[1,i], traj[2,i], str(text[i]), fontsize=5)

    ax.set_xlabel('East',fontsize=40,linespacing=100)
    ax.set_ylabel('North',fontsize=40,linespacing=3.2)
    ax.set_zlabel('Up',fontsize=40,linespacing=3.2)
    ax.view_init(elev=30,azim=-50)

    cbar = plt.colorbar(sc,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=40)
    plt.title(title, fontsize=50)
    plt.show()

def error_boxplot(err, labels=[], title=None, ax=None, show_outliers=False, output_dir=''):
    assert len(labels) == len(err), "The length of labels should be consistent with the length of the err vector"

    if ax is None:
        fig, ax = plt.subplots(figsize=(50,15),sharey=True)
    
    ax.boxplot(err, labels=labels, showfliers=show_outliers)
    
    if title is not None:
        ax.set_title(title)
        plt.savefig(output_dir+title)
    else:
        plt.savefig(output_dir+'error_boxplot.png')
    return ax

def error_histogram(*errs, num_cams=2, labels=[], title=None, ax=None, xlim=40, ylim=350, bin_width=0.1):
    assert len(labels) == len(errs), "The length of labels should be consistent with the length of the err vector"

    if ax is None:
        fig, ax = plt.subplots(sharex=True, sharey=True, dpi=300)
    
    # if xlim > 100:
    #     bins = np.arange(0,100,1)
    #     bins[-1] = xlim
    # else:
    #     bins = np.arange(0,np.ceil(xlim),1)

    # for err, label in zip(errs,labels):
    #     print(label)
    #     ax.hist(err, bins=bins, label=label, alpha=0.5)
    #     # if ax.get_ylim()[-1] < np.max(n):
    #     #     ax.set_ylim([0,np.max(n)+0.75])

    bins = np.arange(0, xlim+1, 1)
    
    for i, (err, label) in enumerate(zip(errs, labels)):
        h, _ = np.histogram(np.clip(err, bins[0], bins[-1]), bins=bins)
        ax.bar(bins[:-1]+(i-1)*bin_width, h, bin_width, label=label, align='center')
    # h, bins, patches = ax.hist([np.clip(err, bins[0], bins[-1]) for err in errs], bins=bins, range=(0,80), label=labels, alpha=.8)
    # ax.set_ylim([0,0.8])


    # xlabels = bins[1:].astype(str)
    # xlabels[-1] += '+'
    # ax.set_xlim([0,xlim])
    # ax.set_xticks(np.arange(len(xlabels))+0.5)
    # ax.set_xticklabels(xlabels)

    # ax.set_xlim([0, xlim])
    # loc = ax.get_xticks()
    # xlabels = ["{:.0f}".format(x) for x in loc]
    # loc[-1] = xlim+0.5
    # xlabels[-1] = "{:.0f}+".format(xlim)
    # xlabels.append(str(xlim)+'+')
    # ax.set_xticks(loc)
    # ax.set_xticklabels(xlabels)
    
    if title is not None:
        ax.set_title(title)
    
    ax.legend()
    
    return ax, bins[:-1]
    


def draw_detection_matches(img1, d1, img2, d2, title='detection_mathches.png', output_dir=''):
    '''
    Function:
        Draw the corresponding detections in the camera views
    Input:
        img1, img2 = two images
        d1, d2 = the matched detections
    '''
    fig = plt.figure()
    dp1 = [cv2.KeyPoint(d[1], d[2], 8) for d in d1.T]
    dp2 = [cv2.KeyPoint(d[1], d[2], 8) for d in d2.T]
    # print(dp1)
    # print(dp2)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(dp1))]

    outimg = cv2.drawMatches(img1, dp1, img2, dp2, matches, None)

    plt.imshow(outimg), plt.show()
    cv2.imwrite(output_dir+title, outimg)    

def draw_matches(img1, kp1, img2, kp2, matches, matchesMask):
    '''
    Function:
        Draw the matching results (FLANN) of the two sets of keypoints
    Input:
        img1, img2 = two images
        kp1, kp2 = two matched keypoints
        matches = the matcher object returned from FLANN matcher
        matchesMask = index of good matches
    '''
    fig = plt.figure()
    draw_params = dict(matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=matchesMask,
                    flags=cv2.DrawMatchesFlags_DEFAULT)
    out_img1 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imwrite('sift_match.png', out_img1)
    plt.imshow(out_img1), plt.show()


if __name__ == "__main__":
    # # Synthetic trajectory data
    # X = np.loadtxt('./data/Synthetic_Trajectory_generated.txt')
    # X_homo = np.insert(X,3,1,axis=0)
    
    # show_trajectory_3D(X,color=True)

    # # Fixposition data
    # X1 = np.loadtxt('./data/fixposition_1_kml.txt',delimiter=',')
    # X1 = X1.T

    # X2 = np.loadtxt('./data/fixposition_2_kml.txt',delimiter=',')
    # X2 = X2.T
    # show_trajectory_3D(X1,X2)

    # Triangulated real trajectory
    with open('./data/test_trajectory.pickle', 'rb') as file:
        results = pickle.load(file)

    for i in range(len(results['Beta'])):
        traj_1, traj_2 = results['X1'][i], results['X2'][i]
        show_trajectory_3D(traj_1,traj_2,
        title='Shift:{}, estimated beta:{:.3f}, without sync (left), with sync (right)'.format(
            results['shift'][i],results['Beta'][i],))

    fig,ax = plt.subplots(1,2,sharex=True)
    ax[0].plot(results['shift'],results['Beta'])
    ax[1].plot(results['shift'],np.asarray(results['Beta'])-np.asarray(results['shift']))
    ax[0].set_title('Estimation of Beta with different shifts')
    ax[1].set_title('Error of Beta with different shifts')
    ax[0].set_ylabel('Estimated Beta')
    ax[1].set_ylabel('Error of Beta')
    plt.xlabel('Shifts from -10 to 10')
    # fig.suptitle('Threshold 1 = {}, Threshold 2 = {}, Degree of spline = {}, Smooth factor = {}'.format(5,5,3,100))
    plt.show()

    print('finished')