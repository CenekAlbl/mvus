import numpy as np
from matplotlib import pyplot as plt
import util
import epipolar as ep
import synchronization
import common_mp_rs as common
import transformations
import scipy.io as scio
import pickle
import argparse
import copy
import cv2
import visualization as vis
from datetime import datetime
from scipy.optimize import least_squares
from scipy import interpolate
from icp_trans import ICP as icp
import pymap3d as pm
import random
from sklearn.metrics import mean_squared_error
import simplekml

'''----------------Previous results----------------'''
# Load ground truth 
gt = np.loadtxt('data2/Raw_gps/GT_ECEF.txt').T

# ### Convert to ENU and Save to Text
ell_wgs84 = pm.Ellipsoid('wgs84')
gt_ll = np.vstack(pm.ecef2geodetic(gt[0],gt[1],gt[2],ell=ell_wgs84))
gt_enu0 = pm.geodetic2enu(gt_ll[0],gt_ll[1],gt_ll[2],gt_ll[0][-10],gt_ll[1][-10],gt_ll[2][-10],ell=ell_wgs84)
gt_enu1 = np.vstack((gt_enu0[0],gt_enu0[1],gt_enu0[2]))

# fn = 'data2/Raw_gps/GT_ENU.txt'
# header = 'FixPosition F2 GT_ENU'
# np.savetxt(fn,gt_enu1.T,delimiter='  ',header=header)

kml = simplekml.Kml()
style = simplekml.Style()
multipnt = kml.newmultigeometry(name="MultiPoint")
#multipnt.style.labelstyle.scale = 0 # Remove the labels from all the points
#multipnt.style.iconstyle.color = simplekml.Color.red
inds = np.arange(0,gt_ll.shape[1],10)
#multipnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
multipnt.style.linestyle.color = 'ff0000ff'  # Red
multipnt.style.linestyle.width= 10  # pnts

lin_coords = [(row[1], row[0],(row[2]-np.min(gt_ll[2]))) for row in gt_ll.T[inds] ]
multipnt.newlinestring(coords=lin_coords,altitudemode=simplekml.AltitudeMode.relativetoground)
for r in gt_ll.T:
    multipnt.newpoint(coords=[(r[1], r[0],(r[2]-np.min(gt_ll[2])))],
    altitudemode=simplekml.AltitudeMode.relativetoground)
kml.save("data2/gt_ll.kml")


#Show Results or not    
show_results = 0

#Init run parameters
use_F = True
include_K = True
include_d = True
include_b = True
include_alpha = 1
max_iter = 3
use_spline = True
motion = False
smooth_factor = 0.005
include_rs = False
rsbounds = True
rs_init = 0
opt_calibration = True
init_pair = True
opt_init_pair = False
final = 0
sparse = True
calmd = 'mat_cal'
section1 = 3000
section2 = -3000
mode = 'spln'
t_ind = '6k'

if show_results:
    print('\n____________________Computing Results__________________\n')

    #Spln
    #with open('data2/Aug_14_results/fpf2_init_cvspln_b_rs.pkl', 'rb') as file:
    #        spln_0_cv = pickle.load(file)

    with open('data2/Aug_14_results/fpf2_ini_mat_cal_spln_b_rs.pkl', 'rb') as file:
            spln_0_mat = pickle.load(file)

    with open('data2/Aug_14_results/fpf2_3cam_spline_cal_s_0.005_mat_cal.pkl', 'rb') as file:
            spln_3_mat = pickle.load(file)

    with open('data2/Aug_14_results/fpf2_4cam_spline_cal_s_0.005_mat_cal.pkl', 'rb') as file:
            spln_4_mat = pickle.load(file)

    with open('data2/Aug_14_results/fpf2_5cam_spline_cal_s_0.005_mat_cal.pkl', 'rb') as file:
            spln_5_mat = pickle.load(file)

    with open('data2/Aug_14_results/fpf2_6cams_spline_cal_s_0.005_mat_cal.pkl', 'rb') as file:
            spln_5_mat = pickle.load(file)

    #with open('data2/Aug_14_results/fp_f2_5camspline_cal_rs_b.pkl', 'rb') as file:
    #        spln_5_cvcal = pickle.load(file)
    
    with open('data2/Aug_14_results/fpf2_4cam_spline_cal_sfct_0.001_mat_cal.pkl', 'rb') as file:
            spln_4_mat = pickle.load(file)
    
    with open('data2/Aug_14_results/fpf2_6cams_spline_cal_sfct_0.001_mat_cal.pkl', 'rb') as file:
            spln_6_mat = pickle.load(file)
        


    bounds = np.all(spln_6_mat.traj[1:,:] < 1,axis=0)

    def filter_traj(traj,cams,err_filt):
        
        for i in range(cams):
            e,ind = traj.error_cam(i)
            if i == 0:
                err_ind = np.array([ind[e<err_filt]])
            else:
                err_ind = np.concatenate((err_ind,ind))
            return err_ind

    err_ind = filter_traj(spln_6_mat,6,30)

    
            

    #vis.show_trajectory_3D(spln_6_mat.traj[1:,bounds]) 

    # load GT_ENU
    gt_enu = np.loadtxt('data2/Raw_gps/GT_ENU.txt').T#
    traj_2 = gt_enu[:,:]

    # Load previous computed flight
    #with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/flight_1_c4_spline.pkl', 'rb') as file:
    #    flight_pre = pickle.load(file)

    #with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/RS_Test/flight_1_idx_testspline_cal.pkl', 'rb') as file:
     #   flight_spln_1 = pickle.load(file)

    # Sample of Optimized Traj.
    # tck,u = interpolate.splprep(traj_mp_vis[1:],u=traj_mp_vis[0],s=0,k=1)
    # x,y,z = interpolate.splev(traj_pre[0],tck)
    # traj_mp_vis = np.array([x,y,z])

    #vis.show_trajectory_3D(traj_pre[1:],traj_mp_vis,line=False,title='Motion Prior Trajectory Reconstruction vs GPS (1st flight)')

    # tck,u = interpolate.splprep(traj_mp_vis_0[1:],u=traj_mp_vis_0[0],s=0,k=1)
    # x,y,z = interpolate.splev(traj_pre[0],tck)
    # traj_mp_vis = np.array([x,y,z])

    #with open('data/GPS_to_reconstruction/Motion_Prior_Test/flight_1_raw_1_MP.pkl', 'rb') as file:
    #with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/flight_1_motion_prior.pkl', 'rb') as file:
    #with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/flight_1_spline_test_sf_001.pkl', 'rb') as file:
    #with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/flight_1_spln_rpjct_err.pkl', 'rb') as file:
    #with open('data/GPS_to_reconstruction/flight_1.pkl', 'rb') as file:
        #flight_pre = pickle.load(file)

    #with open('data/GPS_to_reconstruction/Motion_Prior_Test/errors/flight_1_rpjct_err_raw_0.0_MP.pkl', 'rb') as rpj:
    #    r_err = pickle.load(rpj)


    # Dense Resample of Trajectories


    m = 10
    #traj_1 = flight_pre.traj[1:,649:]
    
    #traj_1 = traj_pre[1:]
    traj_1 = traj_mp_vis[:]
    traj_2 = gt_enu[:,3624:4048]
    #traj_2ll = gt_ll[:,3624:4048]
    #traj_2 = gt_enu[:,3623:4048]

    # Dense Sample of Optimized Traj.    
    tck,u = interpolate.splprep(traj_1,u=np.arange(traj_1.shape[1]),s=0,k=1)
    idx_1 = np.linspace(u[0],u[-1],m*traj_1.shape[1])
    x,y,z = interpolate.splev(idx_1,tck)
    traj_1 = np.array([x,y,z])
    
    
    idx_2 = np.arange(0,6*m*traj_2.shape[1],6*m)
    error_min = 100

    for i in range(traj_1.shape[1]-idx_2[-1]-1):
        #idx_3 = np.sort(np.random.uniform(0,traj_1.shape[1],424))
        idx_3 = idx_1[idx_2+i]
        x,y,z = interpolate.splev(idx_3,tck)
        traj_3 = np.array([x,y,z])

        # Estimate a similarity transformation to align trajectories
        M = transformations.affine_matrix_from_points(traj_3,traj_2,shear=False,scale=True)
        traj_4 = np.dot(M,util.homogeneous(traj_3))
        traj_4 /= traj_4[-1]

        # Evaluation
        scale, shear, angles, translate, perspective = transformations.decompose_matrix(M)
        error = np.sqrt((traj_2[0]-traj_4[0])**2 + (traj_2[1]-traj_4[1])**2 + (traj_2[2]-traj_4[2])**2)
        if np.mean(error) < error_min:
            error_min = np.mean(error)
            k = i
            M_best = M
            traj_4_best = traj_4
            traj_3_best = traj_3

    #print('Mean error between transformed reconstruction and GPS data: {:.5f}, unit is meter.'.format(error_min))
    #vis.show_trajectory_3D(traj_2,traj_4_best,line=False,title='Raw Reconstruction vs GPS (1st flight)')
    #vis.show_trajectory_3D(traj_4_best,traj_2,line=False,title='Reconstruction downsampled and transformed(similarity) vs GPS (1st flight)')

    # tck,u = interpolate.splprep(traj_1,u=np.arange(traj_1.shape[1]),s=0,k=1)
    # #x,y,z = interpolate.splev(np.linspace(0,1,traj_2.shape[1]),tck)
    # x,y,z = interpolate.splev(np.linspace(u[0],u[-1],traj_2.shape[1]),tck)
    # traj_3 = np.array([x,y,z])

    #traj_3hmg = util.homogeneous(traj_3.copy()).T
    traj_2hmg = util.homogeneous(traj_2)
    traj_1hmg = util.homogeneous(traj_1)

    #scale trajectory
    #traj_3hmg[0] = util.mapminmax(traj_3[0],min(traj_2[0]),max(traj_2[0]))
    #traj_3hmg[1] = util.mapminmax(traj_3[0],min(traj_2[1]),max(traj_2[1]))
    #traj_3hmg[2] = util.mapminmax(traj_3[0],min(traj_2[2]),max(traj_2[2]))

    # Estimate a similarity transformation to align trajectories

    #traj_5[0] = traj_5[1]
    #traj_5[1] = traj_5[0]
    #scale trajectory
    #traj_5[2] = -1*traj_5[2] #util.mapminmax(traj_5[0],min(traj_2[0]),max(traj_2[0]))
    # traj_5[1] = util.mapminmax(traj_5[0],min(traj_2[1]),max(traj_2[1]))
    # traj_5[2] = util.mapminmax(traj_5[0],min(traj_2[2]),max(traj_2[2]))

    #traj_5 = traj_5.T

    # M = transformations.affine_matrix_from_points(traj_3,traj_2,shear=False)
    # traj_4 = np.dot(M,util.homogeneous(traj_3))
    # traj_4 /= traj_4[-1]
    # traj_4hmg = traj_4.copy()

    #vis.show_tf_trajectory_3D(traj_4,X_dst=traj_2,color=False,line=False,title='Reconstruction downsampled and transformed(similarity) vs GPS (1st flight)',pause=False)

    #Refine with ICP
    err, Micp, traj_5,scale_icp,dg = icp(traj_4_best,traj_2hmg)
    Micp[:3,:3] #* scale.T

    print('Error (m) reconstruction from detections and GPS data: Mean:{:.5f}/Max:{:.5f}/Median:{:.5f}/RMSE:{:.5f}.\n'.format(
            np.mean(err_icp),np.max(error_icp),
            np.median(err_icp),util.skl_rmse(traj_gt,err_icp[:3])))
    # traj_5 /= traj_5[-1]

    # M = transformations.affine_matrix_from_points(traj_5[:-1,:],traj_2ll,shear=True,scale=True)
    # traj_5ll = np.dot(M,traj_5)
    # traj_5ll /= traj_5ll[-1]

    #M_ll = transformations.affine_matrix_from_points(traj_5[:-1,:] ,traj_2ll,shear=True,scale=False)
    #traj_5ll = np.dot(M_ll,traj_5)

    #traj_5 = traj_5.T

    # Evaluation
    # scale_icp, shear_icp, angles_icp, translate_icp, perspective_icp = transformations.decompose_matrix(Micp)
    # scale, shear, angles, translate, perspective = transformations.decompose_matrix(M)
    error = np.sqrt((traj_2[0]-traj_4_best[0])**2 + (traj_2[1]-traj_4_best[1])**2 + (traj_2[2]-traj_4_best[2])**2)
    print('Mean error between transformed reconstruction and GPS data: {:.6f}, unit is meter.'.format(np.mean(error)))

    error_icp = np.sqrt((traj_2[0]-traj_5[0])**2 + (traj_2[1]-traj_5[1])**2 + (traj_2[2]-traj_5[2])**2)
    print('Mean ICP error between transformed reconstruction and GPS data: {:.6f}, unit is meter.'.format(np.mean(error_icp)))

    #error_icp = np.sqrt((traj_2[0]-traj_5t[0])**2 + (traj_2[1]-traj_5t[1])**2 + (traj_2[2]-traj_5t[2])**2)
    #print('Mean ICP error between transformed reconstruction and GPS data: {:.6f}, unit is meter.'.format(np.mean(error_icp)))

    # Visualization
    vis.show_trajectory_3D(traj_2,traj_4_best,line=False,title='Raw Reconstruction vs GPS (1st flight)',pause=False)
    #vis.show_trajectory_3D(traj_2,traj_5,color=error_icp,line=False,title='Reconstruction downsampled and transformed(similarity) vs GPS (1st flight)',pause=False)
    # #vis.show_tf_trajectory_3D(traj_5ll,X_dst=traj_2ll,color=[],line=False,title='Reconstruction downsampled and transformed(ICP) vs GPS (1st flight)',pause=False)
    vis.show_tf_trajectory_3D(traj_4_best,X_dst=traj_2,color=[],line=False,title='Reconstruction downsampled and transformed(similarity) vs GPS (1st flight)',pause=False)
    vis.show_tf_trajectory_3D(traj_5,X_dst=traj_2,color=[],line=False,title='Reconstruction downsampled and transformed(ICP) vs GPS (1st flight)',pause=False)
    # #vis.show_tf_trajectory_3D(traj_5,X_dst=traj_5t,color=False,line=False,title='Reconstruction downsampled and transformed(ICP) vs GPS (1st flight)',pause=False)
    # #vis.show_tf_trajectory_3D(traj_5,X_dst=traj_2,color=error_icp,line=False,title='Reconstruction downsampled and transformed(ICP) vs GPS (1st flight)',pause=False)

    # '''----Plot Histogram and Error Dist.----'''
    # #with open('./data/fixposition/err_1s.pkl', 'rb') as file:
    # #    err_1s = pickle.load(file)

    plt.figure()
    n_bins = 5
    bins = np.arange(0,30,n_bins)
    colors = ['red', 'lime']
    labels = ['Point_MP', 'Points','Spline']

    statis = np.array([error_icp,error]) * 100
    plt.hist(statis.T, bins, histtype='bar', color=colors, label=labels)

    plt.xticks(bins,fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(loc=1, prop={'size': 30})
    plt.title('Transformed Spline Reconstructed Trajectory vs. GPS Measurements')
    plt.xlabel('Reconstruction Error (cm)')
    plt.ylabel('Number of Points')
    plt.show()

    fig = plt.figure(figsize=(20, 15))
    plt.set_cmap('Wistia')
    ax = fig.add_subplot(111,projection='3d')
    err= error_icp * 100
    sc = ax.scatter3D(traj_5[0],traj_5[1],traj_5[2],c=err,marker='o',s=100)
    ax.set_xlabel('East',fontsize=20)
    ax.set_ylabel('North',fontsize=20)
    ax.set_zlabel('Up',fontsize=20)
    ax.set_title('Transformed Spline Reconstructed Trajectory vs GPS Measurements')
    ax.view_init(elev=30,azim=-50)

    cbar = plt.colorbar(sc,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=40) 
    plt.show()


    fig = plt.figure(figsize=(20, 15))
    plt.set_cmap('Wistia')
    ax = fig.add_subplot(111,projection='3d')
    err= error * 100
    sc = ax.scatter3D(traj_4_best[0],traj_4_best[1],traj_4_best[2],c=err,marker='o',s=100)
    ax.set_xlabel('East',fontsize=20)
    ax.set_ylabel('North',fontsize=20)
    ax.set_zlabel('Up',fontsize=20)
    ax.set_title('Error Distribution (cm) of Transformed Reconstructed Trajectory')
    ax.view_init(elev=30,azim=-50)

    cbar = plt.colorbar(sc,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=40) 
    plt.show()

###########################     '''---------------New computation----------------''' ##########################
###########################
#     # Set parameters manually
if not final:
    
    def filter_traj(traj,cams,err_filt):
        
        for i in range(cams):
            e,ind = traj.error_cam(i)
            if i == 0:
                err_ind = np.array([ind[e<err_filt]])
            else:
                err_ind = np.concatenate((err_ind,ind))
            return err_ind
            
    # # Load camera intrinsics and radial distortions
    # intrin_1 = scio.loadmat('data2/calibration /sony_alpha5n_calib_2pars.mat')
    # intrin_2 = scio.loadmat('data2/calibration /sony_alpha5100_calib_2pars.mat')
    # intrin_3 = scio.loadmat('data2/calibration /sonyG_calib_2pars.mat')
    # intrin_4 = scio.loadmat('data2/calibration /gopro3_calib_2pars.mat')
    # intrin_5 = scio.loadmat('data2/calibration /mate7_calib_2pars.mat')
    # intrin_6 = scio.loadmat('data2/calibration /mate10_calib_2pars.mat')


    with open('data2/calibration_cv2/mate10/mate10_r.pickle', 'rb') as file:
        cal_mate_10 = pickle.load(file)

    with open('data2/calibration_cv2/mate7/mate7_r.pickle', 'rb') as file:
        cal_mate_7 = pickle.load(file)

    with open('data2/calibration_cv2/sonyG/sonyG_r.pickle', 'rb') as file:
        cal_sonyG = pickle.load(file)
    
    with open('data2/calibration_cv2/sony_alpha_5100/sony_alpha_5100_r.pickle', 'rb') as file:
        cal_alpha_5100 = pickle.load(file)
    
    with open('data2/calibration_cv2/sony_alpha_5n/sony_alpha_5n.pickle', 'rb') as file:
        cal_alpha_5n = pickle.load(file)
    
    with open('data2/calibration_cv2/gopro3/gopro3_r.pickle', 'rb') as file:
        cal_gopro3 = pickle.load(file)

    if calmd == 'cv_cal':
        # Intrinsics
        
        K4 = cal_alpha_5n['intrinsic_matrix'] 
        K2 = cal_alpha_5100['intrinsic_matrix'] 
        K3 = cal_sonyG['intrinsic_matrix']
        K5 = cal_gopro3['intrinsic_matrix']
        K1 = cal_mate_7['intrinsic_matrix']
        K0 = cal_mate_10['intrinsic_matrix']

        # Distortion 
        
        d4 = cal_alpha_5n['distortion_coefficients'][0][:2]
        d2 = cal_alpha_5100['distortion_coefficients'][0][:2]
        d3 = cal_sonyG['distortion_coefficients'][0][:2] 
        d5 = cal_gopro3['distortion_coefficients'][0][:2]
        d1 = cal_mate_7['distortion_coefficients'][0][:2]
        d0 = cal_mate_10['distortion_coefficients'][0][:2]
    
    else:
        intrin_4 = scio.loadmat('data2/calib_6cam_mat/calib_sony_a5n.mat')
        intrin_2 = scio.loadmat('data2/calib_6cam_mat/calib_sony_alpha5100.mat')
        intrin_3 = scio.loadmat('data2/calib_6cam_mat/calib_sonyg.mat')
        intrin_5 = scio.loadmat('data2/calib_6cam_mat/calib_gopro3.mat')
        intrin_1 = scio.loadmat('data2/calib_6cam_mat/calib_mate7.mat')
        intrin_0 = scio.loadmat('data2/calib_6cam_mat/calib_mate10.mat')

        K0 = intrin_0['intrinsic']
        K1 = intrin_1['intrinsic']
        K2 = intrin_2['intrinsic']
        K3 = intrin_3['intrinsic']
        K4 = intrin_4['intrinsic']
        K5 = intrin_5['intrinsic']
    
        d0 = intrin_0['radial_distortion'][0] 
        d1  = intrin_1['radial_distortion'][0]
        d2 = intrin_2['radial_distortion'][0]
        d3  = intrin_3['radial_distortion'][0]
        d4  = intrin_4['radial_distortion'][0]
        d5 = intrin_5['radial_distortion'][0]
        

    # Number of Sensor lines for each camera
    n0=  K0[1,2]*2
    n1 = K1[1,2]*2
    n2 = K2[1,2]*2
    n3 = K3[1,2]*2
    n4 = K4[1,2]*2
    n5 = K5[1,2]*2

    # Rolling Shutter Time
    #rs_init = 0

    cameras = [common.Camera(K=K0,d=d0,n=n1,rs=rs_init), common.Camera(K=K1,d=d1,n=n0,rs=rs_init), 
    common.Camera(K=K2,d=d2,n=n2,rs=rs_init), common.Camera(K=K3,d=d3,n=n3,rs=rs_init), 
    common.Camera(K=K4,d=d4,n=n4,rs=rs_init),common.Camera(K=K5,d=d5,n=n3,rs=rs_init)]
    
    #cameras = [common.Camera(K=K1,d=d1), common.Camera(K=K2,d=d2), common.Camera(K=K3,d=d3), common.Camera(K=K4,d=d4)]

    # Load detections
    detect_4 = np.loadtxt('data2/detections/outp_sony_5n1.txt',usecols=(2,0,1)).T
    detect_2 = np.loadtxt('data2/detections/outp_sonyalpha5001.txt',usecols=(2,0,1)).T
    detect_3 = np.loadtxt('data2/detections/outp_sonyg1.txt',usecols=(2,0,1)).T
    detect_5 = np.loadtxt('data2/detections/outp_gopro1.txt',usecols=(2,0,1)).T
    detect_1 = np.loadtxt('data2/detections/outp_mate7_1.txt',usecols=(2,0,1)).T
    detect_0 = np.loadtxt('data2/detections/outp_mate10_1.txt',usecols=(2,0,1)).T
    
    # # Raw Detections
    
    # Add noise    sigma = 'points' #5
    # detect_1[1:] = detect_1[1:] + np.random.randn(2,detect_1.shape[1]) * sigma
    # detect_2[1:] = detect_2[1:] + np.random.randn(2,detect_2.shape[1]) * sigma
    # detect_3[1:] = detect_3[1:] + np.random.randn(2,detect_3.shape[1]) * sigma
    # detect_4[1:] = detect_4[1:] + np.random.randn(2,detect_4.shape[1]) * sigma
    # print('\nNoise added to raw detections: {}'.format(sigma))

    # Create a scene
    flight = common.Scene()
    flight.addCamera(*cameras)

    if section1 is not None and section2 is not None:
        flight.addDetection(detect_0[:,section1:section2], detect_1[:,section1:section2], detect_2[:,section1:section2],
        detect_3[:,section1:section2], detect_4[:,section1:section2], detect_5[:,section1:section2])
    else:
        flight.addDetection(detect_0, detect_1, detect_2,detect_3, detect_4, detect_5)
    
    if include_rs:
        flight.rs = np.array([[flight.cameras[0].rs,flight.cameras[1].rs,flight.cameras[2].rs,
        flight.cameras[3].rs,flight.cameras[4].rs,flight.cameras[5].rs]])

    # Correct radial distortion, can be set to false
    flight.undistort_detections(apply=True)

    # Compute beta for every pair of cameras

    # Compute beta for every pair of cameras
    # flight.beta = np.array([[0,-42,-574.4,-76.4],
    #                         [42,0,-532.4,-34.4],
    #                         [574.4,532.4,0,498],
    #                         [76.4,34.4,498,0]])

    # Betas from visual Inspection 
    #flight.beta = np.array([[0,-550.4756,205.2043,251.1573,-515.4586,-208.8146]])
    
    # Computed Ground Truth
    # Ref alpha_5n
    #flight.beta = np.array([[0,659.9271,248.3173,-602.2104,707.5272,-364.8118]])

    # mate_10 as Reference
    flight.beta = np.array([[0,456.7317,409.5930, -782.45,-208.8148 ,-1102.9]])

    # a5n ref
    #flight.alpha = np.array([[1,1.1988,2.0000,2.3978,1.2010,1.1892]])
    
    # mate10
    flight.alpha = np.array([[1,1.0100,1.0081,1.6819,0.8409,2.0000]])

            #flight.alpha = np.array([[25,29.970030,50,59.940060,30.020690,29.727612]])
            #flight.beta[0] = np.round(flight.beta[0]/flight.alpha,6)


    ##### ------------------------------- Raw Detections ---------------------------------------- #####

    # else:
    #     # Define Beta in Seconds - via visual inspection
    #         #flight.beta = np.array([[0,-0.5,-21,-3]])

    #         flight.alpha = np.array([[29.970030,29.838692,25.000000,25.000000]])
    #         flight.beta = (flight.beta) * flight.alpha[0]/30 #np.array([[0,-0.5,-23,-4]])
        
        # Define Beta in frame indicies
        #flight.beta = flight.beta * flight.alpha 

    #flight.compute_beta(threshold_error=2)

    # create tracks according to beta and alpha
    #if include_alpha:
    #    flight.set_tracks(raw=raw,alpha=flight.alpha[0],spline=[])
    
    flight.set_tracks(auto=True,rs_cor=include_rs)

    # Sort detections in temporal order
    flight.set_sequence()
    flight.set_sequence([0,1,2,3,4,5])
    #flight.set_sequence([5,1,2,3,4,0])

    #flight.compute_beta(d_min=-3,d_max=3)

    #with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/flight_1_beta_raw.pkl'.format(mode),'wb') as f:
    #        pickle.dump(flight, f)

    if use_F:
        E_or_F = 'F'
        error_epip = 10
        error_PnP  = 10
    else:
        E_or_F = 'E'
        error_epip = 0.1
        error_PnP  = 10
    if init_pair:

        if include_d:
            Track = flight.detections
        else:
            Track = flight.detections_undist

        # Initialize the first 3D trajectory
        idx1 = flight.init_traj(error=error_epip,F=use_F,inlier_only=True)

        #print(idx1[:10],idx2[:10])
        # Compute spline parameters and smooth the trajectory
        if use_spline:
            flight.fit_spline(s=smooth_factor)
        else:
            flight.spline = []

    
        init_pair = True
        '''----------------Optimization----------------'''
        start=datetime.now()

        # Record settings
        print('\nCurrently using F for the initial pair, K is optimized, beta and d are optimized, spline not applied')
        print('Threshold for Epipolar:{}, Threshold for PnP:{}'.format(error_epip,error_PnP))

        print('\nBefore optimization:')
        f1,f2,f3,f4,f5,f6 = flight.sequence[0],flight.sequence[1], flight.sequence[2], flight.sequence[3],flight.sequence[4],flight.sequence[5]
        flight.error_cam(f1)
        flight.error_cam(f2)
        flight_before = copy.deepcopy(flight)

        
        '''Optimize two'''

        # Test Spline
        if use_spline:
            res,model = common.optimize_two(flight.cameras[f1],flight.cameras[f2],flight.tracks[f1],
                                flight.tracks[f2],flight.traj,flight.spline,
                                include_K=include_K,motion=False,max_iter=max_iter,calibration=opt_calibration,verbose=2)
            if opt_calibration:
                flight.cameras[f1],flight.cameras[f2],flight.traj = model[0], model[1], model[2]
            else:
                flight.traj = model[0]

            if use_spline:
                flight.spline = model[-1]

            print('\nAfter optimizing first two cameras with spline constraint:')
            flight.error_cam(f1)
            flight.error_cam(f2)

        ## Optimize Detections
        else:
            res_mot,model_mot = common.optimize_two(flight.cameras[f1],flight.cameras[f2],flight.tracks[f1],
                                flight.tracks[f2],flight.traj,spline=[],include_K=include_K,motion=motion,
                                motion_weights=0,max_iter=max_iter,calibration=opt_calibration)

            if opt_calibration:
                flight.cameras[f1],flight.cameras[f2],flight.traj = model_mot[0], model_mot[1], model_mot[2]
            
            else:
                flight.traj = model_mot[0]

            print('\nAfter optimizing first two cameras without spline, Beta, and or RS:')
            flight.error_cam(f1)
            flight.error_cam(f2)
        if use_spline:

            if include_rs:
                mode = 'spln_rs'
            else:
                mode = 'spln'
        else:
            if include_rs:
                mode = 'pnt_rs'
            else:
                mode = 'pnt'

        with open('data2/Aug_14_results/'+str(t_ind)+'_init_{}_{}.pkl'.format(calmd,mode),'wb') as f:
            pickle.dump(flight, f)

        print('\nTime: {}\n'.format(datetime.now()-start))
        
        #init_pair = False
    ''' Optimize Two Cams with Beta and RS params '''
    if opt_init_pair:
        #init_pair = True
        # BA Two Cams & W-RS corrections
        #with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/flight_1_init_pair_{}.pkl'.format(mode), 'rb') as file:
        #    flight_init_pair = pickle.load(file)
        #f1,f2,f3, f4 = flight.sequence[0],flight.sequence[1], flight.sequence[2], flight.sequence[3]
        
        # Fit spline again if needed
        if use_spline:
            flight.fit_spline(s=smooth_factor)

        # Define visibility
        flight.set_visibility()

        '''---------------------   Optimize first 2 cameras with Beta & or RS ------------------'''
        # Before BA: set parameters
        if include_b:
            beta = flight.beta[0,(f1,f2)]
            if include_alpha:
                alpha = flight.alpha[0,(f1,f2)]
            else:
                alpha = []
            if include_d:
                Track = flight.detections
            else:
                Track = flight.detections_undist
        else:
            include_d = False
            beta = []
            Track = flight.tracks
        
        cam_temp = [flight.cameras[f1],flight.cameras[f2]]
        Track_temp = [Track[f1],Track[f2]]
        traj_temp = copy.deepcopy(flight.traj)
        v_temp = np.array([flight.visible[f1],flight.visible[f2]])
        s_temp = copy.deepcopy(flight.spline)

        #Test Spline
        if use_spline:
            res, model = common.optimize_all(cam_temp, Track_temp, traj_temp, v_temp, s_temp,
                                        include_K=include_K,max_iter=max_iter,motion=False,
                                        distortion=include_d,sparse=sparse,calibration=opt_calibration,
                                        rs_cor=include_rs,beta=beta,alpha=alpha,verbose=2,rs_bounds=rsbounds)

            # After BA: interpret results
            if include_b:
                flight.beta[0,(f1,f2)] = model[0]
                if include_rs:
                    flight.cameras[f1].rs,flight.cameras[f2].rs = model[1][0],model[1][1]
                    
                    flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                    flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2] = model[2][0], model[2][1]
                        flight.traj = model[3]
                    else:
                        flight.traj = model[2]
            
                else:
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2] = model[1][0], model[1][1]
                        flight.traj = model[2]
                    else:
                        flight.traj = model[1]

            else:
                if include_rs:
                    flight.cameras[f1].rs,flight.cameras[f2].rs = model[0]
                    
                    flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                    flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2] = model[1][0], model[1][1]
                        flight.traj = model[2]
                    else:
                        flight.traj = model[1]
                else:
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2] = model[0][0], model[0][1]
                        flight.traj = model[1]
                    else:
                        flight.traj = model[0]

            if use_spline:
                flight.spline = model[-1]

            flight.undistort_detections(apply=True)
            #if include_alpha:
            #    flight.set_tracks(raw=raw,alpha=flight.alpha[0])
            #else:
            flight.set_tracks(rs_cor=include_rs)

            # Check reprojection error
            #print('\nAfter optimizing 2 cameras, beta:{}, d:{},spline:{}'.format(include_b,include_d,use_spline))
            #flight.error_cam(f1)
            #flight.error_cam(f2)
            
        # Test Points/Motion Prior
        else:
            res, model = common.optimize_all(cam_temp, Track_temp, traj_temp, v_temp, spline=[],
                                            include_K=include_K,max_iter=max_iter,sparse=sparse,motion=motion,
                                            motion_weights=0,distortion=include_d,beta=beta,alpha=alpha,
                                            calibration=opt_calibration,rs_cor=include_rs,rs_bounds=rsbounds,verbose=2)
            if include_b:
                flight.beta[0,(f1,f2)] = model[0]
                if include_rs:
                    flight.cameras[f1].rs,flight.cameras[f2].rs = model[1][0],model[1][1]
                    
                    flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                    flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2] = model[2][0], model[2][1]
                        flight.traj = model[3]
                    else:
                        flight.traj = model[2]

                elif opt_calibration:
                    flight.cameras[f1],flight.cameras[f2] = model[1][0], model[1][1]
                    flight.traj = model[2]
                else:
                    flight.traj = model[1]
            else:
                if include_rs:
                    flight.cameras[f1].rs,flight.cameras[f2].rs = model[0][0], model[0][1]
                    
                    flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                    flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    

                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2] = model[1][0], model[1][1]
                        flight.traj = model[2]
                    else:
                        flight.traj = model[1]
                else:
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2] = model[0][0], model[0][1]
                        flight.traj = model[1]
                    else:
                        flight.traj = model[0]
                    

            if use_spline:
                flight.spline = model[-1]

            flight.undistort_detections(apply=True)

            if len(alpha):
                flight.set_tracks(raw=False,alpha=flight.alpha[0])
            else:
                flight.set_tracks(rs_cor=include_rs)

        # Check reprojection error
        print('\nAfter optimizing 2 cameras, beta:{}, d:{},spline:{},motion:{}'.format(include_b,include_d,use_spline,motion))
        flight.error_cam(f1)
        flight.error_cam(f2)
        
        print('\nTime: {}\n'.format(datetime.now()-start))
        

    ### Add Third Camera
    if not init_pair:
        with open('data2/Aug_14_results/fpf2_init_{}_{}.pkl'.format(calmd,mode),'rb') as f:
                flight = pickle.load(f)
        f1,f2,f3,f4,f5,f6 = flight.sequence[0],flight.sequence[1], flight.sequence[2], flight.sequence[3],flight.sequence[4],flight.sequence[5]
        init_pair = True

    if init_pair:

        start=datetime.now()
        #with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/flight_1_init_pair.pkl', 'rb') as file:
        #    flight_init_pair = pickle.load(file)
        
        #f1,f2,f3,f4 = flight.sequence[0],flight.sequence[1], flight.sequence[2], flight.sequence[3]
        '''Add the third camera'''
        flight.get_camera_pose(f3,error=error_PnP)
        #flight.get_camera_pose(f3,error=error_PnP,s=0.5,k=3)
        #flight.get_camera_pose(f3,error=error_PnP,s=5,k=2)
        flight.error_cam(f3)

        # Triangulate more points if possible
        flight.triangulate_traj(f1,f3)
        flight.triangulate_traj(f2,f3)

        # Fit spline again if needed
        if use_spline:
            flight.fit_spline(s=smooth_factor)

        print('\n Before Optimizing 3 Cameras:')
        flight.error_cam(f1)
        flight.error_cam(f2)
        flight.error_cam(f3)

        # Define visibility
        flight.set_visibility()

        '''Optimize all 3 cameras'''
        # Before BA: set parameters
        if include_b:
            beta = flight.beta[0,(f1,f2,f3)]
            if include_alpha:
                alpha = flight.alpha[0,(f1,f2,f3)]
            else:
                alpha = []
            if include_d:
                Track = flight.detections
            else:
                Track = flight.detections_undist
        else:
            include_d = False
            beta = []
            Track = flight.tracks

        # BA
        
        cam_temp = [flight.cameras[f1],flight.cameras[f2],flight.cameras[f3]]
        Track_temp = [Track[f1],Track[f2],Track[f3]]
        traj_temp = copy.deepcopy(flight.traj)
        v_temp = np.array([flight.visible[f1],flight.visible[f2],flight.visible[f3]])
        s_temp = copy.deepcopy(flight.spline)

        #Test Spline
        if use_spline:
            res, model = common.optimize_all(cam_temp, Track_temp, traj_temp, v_temp, s_temp,
                                        include_K=include_K,max_iter=max_iter,motion=False,
                                        distortion=include_d,sparse=sparse,calibration=opt_calibration,
                                        rs_cor=include_rs,beta=beta,alpha=alpha,verbose=2,rs_bounds=rsbounds)

            # After BA: interpret results
            if include_b:
                flight.beta[0,(f1,f2,f3)] = model[0]
                if include_rs:
                    flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs = model[1][0],model[1][1],model[1][2]
                    
                    flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                    flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3] = model[2][0], model[2][1], model[2][2]
                        flight.traj = model[3]
                    else:
                        flight.traj = model[2]
            
                else:
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3] = model[1][0], model[1][1], model[1][2]
                        flight.traj = model[2]
                    else:
                        flight.traj = model[1]

            else:
                if include_rs:
                    flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs = model[0][0],model[0][1],model[0][2]
                    
                    flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                    flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3] = model[1][0], model[1][1], model[1][2]
                        flight.traj = model[2]
                    else:
                        flight.traj = model[1]
                else:
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3] = model[0][0], model[0][1], model[0][2]
                        flight.traj = model[1]
                    else:
                        flight.traj = model[0]

            if use_spline:
                flight.spline = model[-1]

            flight.undistort_detections(apply=True)
            #if include_alpha:
            #    flight.set_tracks(raw=raw,alpha=flight.alpha[0])
            #else:
            flight.set_tracks(rs_cor=include_rs)

            # Check reprojection error
            print('\nAfter optimizing 3 cameras, beta:{}, d:{},spline:{}'.format(include_b,include_d,use_spline))
            flight.error_cam(f1)
            flight.error_cam(f2)
            flight.error_cam(f3)

        # Test Points/Motion Prior
        else:
            
            res, model = common.optimize_all(cam_temp, Track_temp, traj_temp, v_temp, spline=[],
                                            include_K=include_K,max_iter=max_iter,motion=motion,motion_weights=0,
                                            distortion=include_d,beta=beta,alpha=alpha,calibration=opt_calibration,
                                            sparse=sparse,opt_beta=True,rs_cor=include_rs,rs_bounds=rsbounds,verbose=2)
            if include_b:
                flight.beta[0,(f1,f2,f3)] = model[0]
                if include_rs:
                    flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs = model[1][0],model[1][1],model[1][2]
                    
                    flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                    flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3] = model[2][0], model[2][1], model[2][2]
                        flight.traj = model[3]
                    else:
                        flight.traj = model[2]

                elif opt_calibration:
                    flight.cameras[f1],flight.cameras[f2],flight.cameras[f3] = model[1][0], model[1][1], model[1][2]
                    flight.traj = model[2]
                else:
                    flight.traj = model[1]
            else:
                if include_rs:
                    flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs = model[0][0],model[0][1],model[0][2]
                    
                    flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                    flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    

                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3] = model[1][0], model[1][1], model[1][2]
                        flight.traj = model[2]
                    else:
                        flight.traj = model[1]
                else:
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3] = model[0][0], model[0][1], model[0][2]
                        flight.traj = model[1]
                    else:
                        flight.traj = model[0]
                    

            if use_spline:
                flight.spline = model[-1]

            flight.undistort_detections(apply=True)

            flight.set_tracks(rs_cor=include_rs)

            # Check reprojection error
            print('\nAfter optimizing 3 cameras, beta:{}, d:{},spline:{},motion:{}'.format(include_b,include_d,use_spline,motion))
            flight.error_cam(f1)
            flight.error_cam(f2)
            flight.error_cam(f3)

        

        print('\nTime: {}\n'.format(datetime.now()-start))

        if use_spline:
            if include_K:
                if include_rs:
                    if rsbounds:
                        opt_method = 'spline_cal_rs_b' #{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'spline_cal_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'spline_cal_s_{}'.format(smooth_factor)
            else:
                if include_rs:
                    if rsbounds:
                        opt_method = 'spline_rs_b' #_{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'spline_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'spline_s_{}'.format(smooth_factor)
        else:
            if include_K:
                if include_rs:
                    if rsbounds:
                        opt_method = 'points_cal_rs_b'# _{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'points_cal_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                    
                else:
                    opt_method = 'points_cal'
            else:
                if include_rs:
                    if rsbounds:
                        opt_method = 'points_rs_b' #_{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'points_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'points'
        print('Finished:{}'.format(opt_method))
        opt_method += '_' + str(calmd)
        with open('data2/Aug_14_results/'+str(t_ind)+'_3_'+str(opt_method)+'.pkl'.format(calmd),'wb') as f:
            pickle.dump(flight, f)

        '''Add the fourth camera'''
        flight.get_camera_pose(f4,error=error_PnP)
        flight.error_cam(f4)

        # Triangulate more points if possible
        flight.triangulate_traj(f1,f4)
        flight.triangulate_traj(f2,f4)
        flight.triangulate_traj(f3,f4)

        #sigma = '4_cam_pnts'
        #with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/flight_1_'+str(sigma)+'.pkl','wb') as f:
        #    pickle.dump(flight, f)

        # Fit spline again if needed
        if use_spline:
            flight.fit_spline(s=smooth_factor)

        flight.error_cam(f1)
        flight.error_cam(f2)
        flight.error_cam(f3)
        flight.error_cam(f4)

        # Define visibility
        flight.set_visibility()

        '''Optimize all 4 cameras'''
        # Before BA: set parameters
        if include_b:
            beta = flight.beta[0,(f1,f2,f3,f4)]
            if include_alpha:
                alpha = flight.alpha[0,(f1,f2,f3,f4)]
            else:
                alpha = []
            if include_d:
                Track = flight.detections
            else:
                Track = flight.detections_undist
        else:
            include_d = False
            beta = []
            Track = flight.tracks
        
        # BA
        
        cam_temp = [flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4]]
        Track_temp = [Track[f1],Track[f2],Track[f3],Track[f4]]
        traj_temp = copy.deepcopy(flight.traj)
        v_temp = np.array([flight.visible[f1],flight.visible[f2],flight.visible[f3],flight.visible[f4]])
        s_temp = copy.deepcopy(flight.spline)

        # BA
        ######## ----------------------------------- Test Spline ------------------------------------------#######
        if use_spline:
            res, model = common.optimize_all(cam_temp,Track_temp,traj_temp,v_temp,
            s_temp,include_K=include_K,max_iter=max_iter,motion=False,
            calibration=opt_calibration,rs_cor=include_rs,distortion=include_d,
            sparse=sparse,beta=beta,alpha=alpha,verbose=2,rs_bounds=rsbounds)

            # After BA: interpret result            
            if include_b:
                if include_rs:
                    if opt_calibration:
                        flight.beta[0,(f1,f2,f3,f4)], flight.traj = model[0],model[3]
                        
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f3]  = model[2][0], model[2][1], model[2][2],model[2][3]

                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs = model[1][0],model[1][1],model[1][2],model[1][3]
                        
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    else:
                        flight.beta[0,(f1,f2,f3,f4)], flight.traj = model[0], model[2]

                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs = model[1][0],model[1][1],model[1][2],model[1][3]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                else:
                    if opt_calibration:
                        flight.beta[0,(f1,f2,f3,f4)],flight.traj = model[0], model[2]
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f3]  = model[1][0], model[1][1], model[1][2],model[1][3]
                    else:
                        flight.beta[0,(f1,f2,f3,f4)], flight.traj = model[0], model[1]
                    
            else:
                if include_rs:
                    if opt_calibration:
                    
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs = model[0][0],model[0][1],model[0][2],model[0][3]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                        
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4]  = model[1][0], model[1][1], model[1][2],model[1][3]
                        
                        flight.traj = model[2]    

                    else:
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs = model[0][0],model[0][1],model[0][2],model[0][3]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                        flight.traj = model[1]
                else:
                    if opt_calibration:
                        
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4]  = model[0][0], model[0][1], model[0][2],model[0][3]
                        flight.traj =  model[1]
                    else:
                        flight.traj = model[0]

            if use_spline:
                flight.spline = model[-1]

            flight.undistort_detections(apply=True)
            #if include_alpha:
            #    flight.set_tracks(raw=raw,alpha=flight.alpha[0])
            flight.set_tracks(rs_cor=include_rs)

            # Check reprojection error
            print('\nAfter optimizing 4 cameras, beta:{}, d:{},spline:{}'.format(include_b,include_d,use_spline))
            flight.error_cam(f1)
            flight.error_cam(f2)
            flight.error_cam(f3)
            flight.error_cam(f4)
        
        ############## ------------------------------- Test Motion Prior ---------------------- #################
        else:
            spline = []
            res_mot, model_mot = common.optimize_all(cam_temp,Track_temp,traj_temp,v_temp,s_temp,include_K=include_K,
                                    max_iter=max_iter,motion=motion,calibration=opt_calibration,sparse=sparse,
                                    rs_cor=include_rs,distortion=include_d,beta=beta,alpha=alpha,verbose=2,rs_bounds=rsbounds)

            # After Points BA: interpret results
            if include_b:
                #if include_alpha:
                #    flight.beta[0], flight.alpha[0], flight.cameras, flight.traj = model_mot[0], model_mot[1], model_mot[2], model_mot[3]
                if include_rs:
                    if opt_calibration:
                        flight.beta[0,(f1,f2,f3,f4)],  flight.traj = model_mot[0], model_mot[3]
                        
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4]  = model_mot[2][0], model_mot[2][1], model_mot[2][2],model[2][3]
                        
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs = model_mot[1][0],model_mot[1][1],model_mot[1][2],model_mot[1][3]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    else:
                        flight.beta[0,(f1,f2,f3,f4)],flight.traj = model_mot[0], model_mot[2]
                        
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs = model_mot[1][0],model_mot[1][1],model_mot[1][2],model_mot[1][3]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                else:
                    if opt_calibration:
                        flight.beta[0,(f1,f2,f3,f4)],flight.traj = model_mot[0], model_mot[2]
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4] = model_mot[1][0],model_mot[1][1],model_mot[1][2],model_mot[1][3]
                    else:
                        flight.beta[0,(f1,f2,f3,f4)], flight.traj = model_mot[0], model_mot[1]
            
            else:
                if include_rs:
                    if opt_calibration:
                        
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs = model_mot[0][0],model_mot[0][1],model_mot[0][2],model_mot[0][3]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                        
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4] = model_mot[1][0],model_mot[1][1],model_mot[1][2],model_mot[1][3]
                        flight.traj =  model_mot[2]

                        

                    else:
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs = model_mot[0][0],model_mot[0][1],model_mot[0][2],model_mot[0][3]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                        
                        flight.traj =  model_mot[1]
                else:
                    if opt_calibration:
                        
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4]  = model_mot[0][0], model_mot[0][1], model_mot[0][2],model[0][3]
                        
                        flight.traj = model_mot[1]
                    else:
                        flight.traj = model_mot[0]

            if use_spline:
                flight.spline = model[-1]

            flight.undistort_detections(apply=True)
            #if len(alpha):
            #    flight.set_tracks(raw=raw,alpha=flight.alpha[0],spline=use_spline)
            
            flight.set_tracks(rs_cor=include_rs)

            # Check reprojection error
            print('\nAfter optimizing 4 cameras, beta:{}, d:{},spline:{},motion:{}'.format(include_b,include_d,use_spline,motion))
            flight.error_cam(f1)
            flight.error_cam(f2)
            flight.error_cam(f3)
            flight.error_cam(f4)


        print('\nTime: {}\n'.format(datetime.now()-start))

        if use_spline:
            if include_K:
                if include_rs:
                    if rsbounds:
                        opt_method = 'spline_cal_rs_b' #{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'spline_cal_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'spline_cal_s_{}'.format(smooth_factor)
            else:
                if include_rs:
                    if rsbounds:
                        opt_method = 'spline_rs_b' #_{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'spline_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'spline_s_{}'.format(smooth_factor)
        else:
            if include_K:
                if include_rs:
                    if rsbounds:
                        opt_method = 'points_cal_rs_b'# _{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'points_cal_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                    
                else:
                    opt_method = 'points_cal'
            else:
                if include_rs:
                    if rsbounds:
                        opt_method = 'points_rs_b' #_{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'points_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'points'
        
        print('Finished:{}'.format(opt_method))
        opt_method += '_' + str(calmd)
        with open('data2/Aug_14_results/'+str(t_ind)+'_4_'+str(opt_method)+'.pkl'.format(calmd),'wb') as f:
            pickle.dump(flight, f)
        #final = 0

        '''Add the fifth camera'''
        flight.get_camera_pose(f5,error=error_PnP)
        flight.error_cam(f5)

        # Triangulate more points if possible
        flight.triangulate_traj(f1,f5)
        flight.triangulate_traj(f2,f5)
        flight.triangulate_traj(f3,f5)
        flight.triangulate_traj(f4,f5)

        #sigma = '4_cam_pnts'
        #with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/flight_1_'+str(sigma)+'.pkl','wb') as f:
        #    pickle.dump(flight, f)

        # Fit spline again if needed
        if use_spline:
            flight.fit_spline(s=smooth_factor)

        flight.error_cam(f1)
        flight.error_cam(f2)
        flight.error_cam(f3)
        flight.error_cam(f4)
        flight.error_cam(f5)

        # Define visibility
        flight.set_visibility()

        '''Optimize 5 cameras'''
        # Before BA: set parameters
        if include_b:
            beta = flight.beta[0,(f1,f2,f3,f4,f5)]
            if include_alpha:
                alpha = flight.alpha[0,(f1,f2,f3,f4,f5)]
            else:
                alpha = []
            if include_d:
                Track = flight.detections
            else:
                Track = flight.detections_undist
        else:
            include_d = False
            beta = []
            Track = flight.tracks

        cam_temp = [flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5]]
        Track_temp = [Track[f1],Track[f2],Track[f3],Track[f4],Track[f5]]
        traj_temp = copy.deepcopy(flight.traj)
        v_temp = np.array([flight.visible[f1],flight.visible[f2],flight.visible[f3],flight.visible[f4],flight.visible[f5]])
        s_temp = copy.deepcopy(flight.spline)
        
        # BA
        ######## ----------------------------------- Test Spline ------------------------------------------#######
        if use_spline:
            res, model = common.optimize_all(cam_temp,Track_temp,traj_temp,v_temp,s_temp,include_K=include_K,
            max_iter=max_iter,motion=False,calibration=opt_calibration,rs_cor=include_rs,distortion=include_d,
            sparse=sparse,beta=beta,alpha=alpha,verbose=2,rs_bounds=rsbounds)

            # After BA: interpret results
            if include_b:
                if include_rs:
                    if opt_calibration:
                        flight.beta[0,(f1,f2,f3,f4,f5)],  flight.traj = model[0], model[3]

                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5]  = model[2][0], model[2][1], model[2][2],model[2][3],model[2][4]
                        
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs= model[1][0],model[1][1],model[1][2],model[1][3],model[1][4]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    else:
                        flight.beta[0,(f1,f2,f3,f4,f5)], flight.traj = model[0], model[2]

                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs = model[1][0],
                        model[1][1],model[1][2],model[1][3],model[1][4]
                        
                        flight.rs = np.array([[flight.cameras[0].rs,flight.cameras[1].rs,flight.cameras[2].rs,
                        flight.cameras[3].rs,flight.cameras[4].rs,flight.cameras[5].rs]])   

                else:
                    if opt_calibration:
                        flight.beta[0,(f1,f2,f3,f4,f5)], flight.traj = model[0],  model[2]
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5]  = model[1][0], model[1][1], model[1][2],model[1][3],model[1][4]
                    else:
                        flight.beta[0,(f1,f2,f3,f4,f5)], flight.traj = model[0], model[1]
                    
            else:
                if include_rs:
                    if opt_calibration:
                    
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs= model[0][0],model[0][1],
                        model[0][2],model[0][3],model[0][4]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5]  = model[1][0], model[1][1], model[1][2],model[1][3],model[1][4]
                        flight.traj = model[2]    

                    else:
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs= model[0][0],model[0][1],
                        model[0][2],model[0][3],model[0][4]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                        flight.traj = model[1]
                else:
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5]  = model[0][0], model[0][1], model[0][2],model[0][3],model[0][4]
                        flight.traj = model[1]
                    else:
                        flight.traj = model[0]

            if use_spline:
                flight.spline = model[-1]

            flight.undistort_detections(apply=True)
            #if include_alpha:
            #    flight.set_tracks(raw=raw,alpha=flight.alpha[0])
            flight.set_tracks(rs_cor=include_rs)

            # Check reprojection error
            print('\nAfter optimizing 5 cameras, beta:{}, d:{},spline:{}'.format(include_b,include_d,use_spline))
            flight.error_cam(f1)
            flight.error_cam(f2)
            flight.error_cam(f3)
            flight.error_cam(f4)
            flight.error_cam(f5)
        
        ############## ------------------------------- Test Motion Prior ---------------------- #################
        #'''' points 5 cameras """
        
        else:
            spline = []
            res_mot, model_mot = common.optimize_all(cam_temp,Track_temp,traj_temp,v_temp,s_temp,include_K=include_K,
                                    max_iter=max_iter,motion=motion,calibration=opt_calibration,sparse=sparse,
                                    rs_cor=include_rs,distortion=include_d,beta=beta,alpha=alpha,verbose=2,rs_bounds=rsbounds)

            ##After Points BA: interpret results
            if include_b:
                #if include_alpha:
                #    flight.beta[0], flight.alpha[0], flight.cameras, flight.traj = model_mot[0], model_mot[1], model_mot[2], model_mot[3]
                if include_rs:
                    if opt_calibration:
                        flight.beta[0,(f1,f2,f3,f4,f5)], flight.traj = model_mot[0], model_mot[3]
                        
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs = model_mot[1][0],model_mot[1][1],model_mot[1][2],model_mot[1][3],model_mot[1][4]
                        
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5]  = model_mot[2][0], model_mot[2][1], model_mot[2][2],model[2][3],model[2][4]

                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                    else:
                        flight.beta[0,(f1,f2,f3,f4,f5)],flight.traj = model_mot[0], model_mot[2]
                        
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs, flight.cameras[f5].rs = model_mot[1][0],model_mot[1][1],
                        model_mot[1][2],model_mot[1][3],model_mot[1][4]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                else:
                    if opt_calibration:
                        flight.beta[0,(f1,f2,f3,f4,f5)], flight.traj = model_mot[0],  model_mot[2]
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5] = model_mot[1][0],model_mot[1][1],model_mot[1][2],model_mot[1][3],model_mot[1][4]
                        
                    else:
                        flight.beta[0,(f1,f2,f3,f4,f5)], flight.traj = model_mot[0], model_mot[1]
            
            else:
                if include_rs:
                    if opt_calibration:
                        

                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs = model_mot[0][0],model_mot[0][1],model_mot[0][2],model_mot[0][3],model_mot[0][4]
                        
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5] = model_mot[1][0],model_mot[1][1],model_mot[1][2],model_mot[1][3],model_mot[1][4]

                        flight.traj =  model_mot[2]

                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                    else:
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs = model_mot[0][0],model_mot[0][1],model_mot[0][2],model_mot[0][3],model_mot[0][4]
                        
                        flight.rs = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                        
                        flight.traj =  model_mot[1]
                else:
                    if opt_calibration:
                        flight.cameras[f1],flight.cameras[f2],flight.cameras[f3],flight.cameras[f4],flight.cameras[f5] = model_mot[0][0],model_mot[0][1],model_mot[0][2],model_mot[0][3],model_mot[0][4]
                        flight.traj =  model_mot[1]
                    else:
                        flight.traj = model_mot[0]

            if use_spline:
                flight.spline = model[-1]

            flight.undistort_detections(apply=True)
            #if len(alpha):
            #    flight.set_tracks(raw=raw,alpha=flight.alpha[0],spline=use_spline)
            
            flight.set_tracks(rs_cor=include_rs)

            # Check reprojection error
            print('\nAfter optimizing 5 cameras, beta:{}, d:{},spline:{},motion:{}'.format(include_b,include_d,use_spline,motion))
            flight.error_cam(f1)
            flight.error_cam(f2)
            flight.error_cam(f3)
            flight.error_cam(f4)
            flight.error_cam(f5)
            
        print('\nTime: {}\n'.format(datetime.now()-start))

        if use_spline:
            if include_K:
                if include_rs:
                    if rsbounds:
                        opt_method = 'spline_cal_rs_b' #{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'spline_cal_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'spline_cal_s_{}'.format(smooth_factor)
            else:
                if include_rs:
                    if rsbounds:
                        opt_method = 'spline_rs_b' #_{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'spline_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'spline_s_{}'.format(smooth_factor)
        else:
            if include_K:
                if include_rs:
                    if rsbounds:
                        opt_method = 'points_cal_rs_b'# _{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'points_cal_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                    
                else:
                    opt_method = 'points_cal'
            else:
                if include_rs:
                    if rsbounds:
                        opt_method = 'points_rs_b' #_{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'points_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'points'

        print('Finished:{}'.format(opt_method))
        opt_method += '_' + str(calmd)
        with open('data2/Aug_14_results/'+str(t_ind)+'_5cam_'+str(opt_method)+'.pkl','wb') as f:
            pickle.dump(flight, f)
        
        '''_____________________________Add the six_th camera__________________________________'''
        flight.get_camera_pose(f6,error=error_PnP)
        flight.error_cam(f6)

        # Triangulate more points if possible
        flight.triangulate_traj(f1,f6)
        flight.triangulate_traj(f2,f6)
        flight.triangulate_traj(f3,f6)
        flight.triangulate_traj(f4,f6)
        flight.triangulate_traj(f5,f6)

        #sigma = '4_cam_pnts'
        #with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/flight_1_'+str(sigma)+'.pkl','wb') as f:
        #    pickle.dump(flight, f)

        # Fit spline again if needed
        if use_spline:
            flight.fit_spline(s=smooth_factor)

        flight.error_cam(f1)
        flight.error_cam(f2)
        flight.error_cam(f3)
        flight.error_cam(f4)
        flight.error_cam(f5)
        flight.error_cam(f6)

        # Define visibility
        flight.set_visibility()

        '''Optimize 6 cameras'''
        # Before BA: set parameters
        if include_b:
            beta = flight.beta[0]
            if include_alpha:
                alpha = flight.alpha[0]
            else:
                alpha = []
            if include_d:
                Track = flight.detections
            else:
                Track = flight.detections_undist
        else:
            include_d = False
            beta = []
            Track = flight.tracks

        
        # BA
        ######## ----------------------------------- Test Spline ------------------------------------------#######
        if use_spline:
            res, model = common.optimize_all(flight.cameras,flight.detections,flight.traj,flight.visible,flight.spline,
            include_K=include_K,max_iter=max_iter,motion=False,
            calibration=opt_calibration,rs_cor=include_rs,distortion=include_d,
            sparse=sparse,beta=beta,alpha=alpha,verbose=2,rs_bounds=rsbounds)

            # After BA: interpret results
            if include_b:
                if include_rs:
                    if opt_calibration:
                        flight.beta[0], flight.cameras, flight.traj = model[0], model[2],model[3]
                        
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs = model[1][0],model[1][1],model[1][2],model[1][3],model[1][4],model[1][5]
                        flight.rs[0] = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])
                    else:
                        flight.beta[0], flight.traj = model[0], model[2]

                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs = model[1][0],model[1][1],model[1][2],model[1][3],model[1][4],model[1][5]
                        flight.rs[0] = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])    

                else:
                    if opt_calibration:
                        flight.beta[0], flight.cameras,flight.traj = model[0], model[1], model[2]
                    else:
                        flight.beta[0], flight.traj = model[0], model[1]
                    
            else:
                if include_rs:
                    if opt_calibration:
                    
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs= model[0][0],model[0][1],
                        model[0][2],model[0][3],model[0][4],model[0][5]
                        
                        flight.rs[0] = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,
                        flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                        flight.cameras, flight.traj = model[1], model[2]    

                    else:
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,
                        flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs= model[0][0],model[0][1],
                        model[0][2],model[0][3],model[0][4],model[0][5]
                        
                        flight.rs[0] = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,
                        flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                        flight.traj = model[1]
                else:
                    if opt_calibration:
                        flight.cameras, flight.traj = model[0], model[1]
                    else:
                        flight.traj = model[0]

            if use_spline:
                flight.spline = model[-1]

            flight.undistort_detections(apply=True)
            #if include_alpha:
            #    flight.set_tracks(raw=raw,alpha=flight.alpha[0])
            flight.set_tracks(rs_cor=include_rs)

            # Check reprojection error
            print('\nAfter optimizing 6 cameras, beta:{}, d:{},spline:{}'.format(include_b,include_d,use_spline))
            flight.error_cam(f1)
            flight.error_cam(f2)
            flight.error_cam(f3)
            flight.error_cam(f4)
            flight.error_cam(f5)
            flight.error_cam(f6)
        
        ############## ------------------------------- Test Motion Prior ---------------------- #################
        ############ _______ points 6 cameras ########################
        else:
            spline = []
            res_mot, model_mot = common.optimize_all(flight.cameras[f1],flight.cameras[f2],flight.tracks[f1],
            flight.tracks[f2],flight.traj,flight.spline,include_K=include_K,max_iter=max_iter,motion=motion,calibration=opt_calibration,sparse=sparse,
            rs_cor=include_rs,distortion=include_d,beta=beta,alpha=alpha,verbose=2,rs_bounds=rsbounds)

            # After Points BA: interpret results
            if include_b:
                #if include_alpha:
                #    flight.beta[0], flight.alpha[0], flight.cameras, flight.traj = model_mot[0], model_mot[1], model_mot[2], model_mot[3]
                if include_rs:
                    if opt_calibration:
                        flight.beta[0], flight.cameras, flight.traj = model_mot[0], model_mot[2],model_mot[3]
                        
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs = model_mot[1][0],model_mot[1][1],model_mot[1][2],model_mot[1][3],model_mot[1][4],model_mot[1][5]
                        flight.rs[0] = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                    else:
                        flight.beta[0],flight.traj = model_mot[0], model_mot[2]
                        
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs= model_mot[1][0],model_mot[1][1],
                        model_mot[1][2],model_mot[1][3],model_mot[1][4],model_mot[1][5]
                        
                        flight.rs[0] = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                else:
                    if opt_calibration:
                        flight.beta[0], flight.cameras,flight.traj = model_mot[0], model_mot[1], model_mot[2]
                    else:
                        flight.beta[0], flight.traj = model_mot[0], model_mot[1]
            
            else:
                if include_rs:
                    if opt_calibration:
                        flight.cameras, flight.traj = model_mot[1], model_mot[2]

                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs = model_mot[0][0],model_mot[0][1],model_mot[0][2],model_mot[0][3],model_mot[0][4],model_mot[0][5]
                        
                        flight.rs[0] = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]])

                    else:
                        flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs = model_mot[0][0],model_mot[0][1],model_mot[0][2],model_mot[0][3],model_mot[1][4],model_mot[1][5]
                        
                        flight.rs[0] = np.array([[flight.cameras[f1].rs,flight.cameras[f2].rs,flight.cameras[f3].rs,flight.cameras[f4].rs,flight.cameras[f5].rs,flight.cameras[f6].rs]]) 
                        
                        flight.traj =  model_mot[1]
                else:
                    if opt_calibration:
                        flight.cameras, flight.traj = model_mot[0], model_mot[1]
                    else:
                        flight.traj = model_mot[0]

            if use_spline:
                flight.spline = model[-1]

            flight.undistort_detections(apply=True)
            #if len(alpha):
            #    flight.set_tracks(raw=raw,alpha=flight.alpha[0],spline=use_spline)
            
            flight.set_tracks(rs_cor=include_rs)

            # Check reprojection error
            print('\nAfter optimizing 6 cameras, beta:{}, d:{},spline:{},motion:{}'.format(include_b,include_d,use_spline,motion))
            flight.error_cam(f1)
            flight.error_cam(f2)
            flight.error_cam(f3)
            flight.error_cam(f4)
            flight.error_cam(f5)
            flight.error_cam(f6)
            


        print('\nTime: {}\n'.format(datetime.now()-start))

        if use_spline:
            if include_K:
                if include_rs:
                    if rsbounds:
                        opt_method = 'spline_cal_rs_b' #{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'spline_cal_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'spline_cal_s_{}'.format(smooth_factor)
            else:
                if include_rs:
                    if rsbounds:
                        opt_method = 'spline_rs_b' #_{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'spline_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'spline_s_{}'.format(smooth_factor)
        else:
            if include_K:
                if include_rs:
                    if rsbounds:
                        opt_method = 'points_cal_rs_b'# _{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'points_cal_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                    
                else:
                    opt_method = 'points_cal'
            else:
                if include_rs:
                    if rsbounds:
                        opt_method = 'points_rs_b' #_{}'.format(np.format_float_scientific(rs_init,3))
                    else:
                        opt_method = 'points_rs' #_{}'.format(np.format_float_scientific(rs_init,3))
                else:
                    opt_method = 'points'
        print('Finished:{}'.format(opt_method))
        opt_method += '_' + str(calmd)
        with open('data2/Aug_14_results/'+str(t_ind)+'_6c_'+str(opt_method)+'.pkl','wb') as f:
            pickle.dump(flight, f)
        final = 0



######
######### ----------------------------------- Final BA w/Raw Detections ------------------------------------------#######
if final:
    with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/RS_Test/Aug_7_L/flight_1_points_rs_b.pkl', 'rb') as file:
        flight = pickle.load(file)
    # Set parameters manually
    use_F = True
    include_K = False
    include_d = True
    include_b = True
    include_alpha = 1
    max_iter = 30
    use_spline = False
    raw = 0
    include_rs = False
    rsbounds = False
    rs_init = 0
    opt_calibration = False
    init_pair = False
    final = 1
    sparse = True
    motion = True
    prior = 'F'
    smooth_factor = 0.005   # 0.005
    rs_cor = True
    opt_beta = False
    # Raw Detections

    detect_1 = np.loadtxt('data/GPS_to_reconstruction/Fixposition_data/Raw detection from Jacky/c1_f1_MOV.txt',usecols=(2,0,1)).T
    detect_2 = np.loadtxt('data/GPS_to_reconstruction/Fixposition_data/Raw detection from Jacky/cam2_flight1_mp4.txt',usecols=(2,0,1)).T
    detect_3 = np.loadtxt('data/GPS_to_reconstruction/Fixposition_data/Raw detection from Jacky/flight_1_MTS.txt',usecols=(2,0,1)).T
    detect_4 = np.loadtxt('data/GPS_to_reconstruction/Fixposition_data/Raw detection from Jacky/c4_f1_MP4.txt',usecols=(2,0,1)).T

    flight.addDetection(detect_1, detect_2, detect_3, detect_4)

    flight.detections = flight.detections[4:]

    # flight.get_camera_pose(f3,error=error_PnP)
    # flight.get_camera_pose(f4,error=error_PnP)
    
    # # Define visibility
    # flight.set_visibility()


    # '''Optimize Final Trajectory'''
    # # Before BA: set parameters

    #flight.beta = np.array([[0,-0.5,-23,-4]])
    
    flight.alpha = np.array([[29.970030,29.838692,25.000000,25.000000]])

    flight.beta = (flight.beta) * flight.alpha[0]/30 
    #np.array([[0,-0.5,-23,-4]])

    #if include_b:
    beta = flight.beta[0]
    alpha = flight.alpha[0]
    
    if include_d:
        Tracks = flight.detections
    else:
        Track = flight.detections_undist
    # else:
    #     include_d = False
    #     beta = []
    #     Track = flight.tracks

    # Correct radial distortion, can be set to false
    flight.undistort_detections(apply=True)

    start=datetime.now()

    ######## ----------------------------------- Test Raw Detections ------------------------------------------#######
    #if motion:
    
    motion_weights = [1000] #[1e-3,1e1,1e2,2.5e3,5e3,1e4,1e5,1e6,1e8,1e10]# np.logspace(2,4,10) #np.hstack((np.logspace(-3,6,2) #, np.logspace(3,8,12)))

    spline = []

    #int_trajs = []

    # Correct for time-scale and time-shift
    
    flight.set_tracks(auto=True,raw=raw,rs_cor=True)
    #Tracks = common.detect_to_track(Track,beta,alpha)

    #flight.tracks = Tracks

    # Interpolate Corresponding 3D points for raw detections from Spline Trajectory
    #spline = flight.fit_spline
    raw_trajs = common.interpolate_track(flight.tracks,flight.traj)

    spline_traj = flight.traj

    f1,f2,f3,f4 = flight.sequence

    # Combine interpolated trajectories from each raw track
    raw_traj = common.combine_traj(raw_trajs) #,[f1,f2,f3,f4])

    #flight.traj = raw_traj

    #flight.set_visibility()

    ## Check for detections
    # vis_traj = np.empty([])
    # for i in range(flight.visible.shape[1]):
    #     if np.sum(flight.visible[:,i]) == 2:
    #         if vis_traj.shape == ():
    #             vis_traj = np.array([raw_traj[:,i]]).T
    #         else:
    #             vis_traj = np.hstack((vis_traj,np.array([raw_traj[:,i]]).T))
    
    #flight.traj = vis_traj
    #flight.set_visibility()

    # Take sample of raw detections
    #sample = np.arange(0,flight.traj.shape[1],5)

    #sample = np.sort(random.sample(range(1, 100), 10))

    #np.random.rand(8454)
    
    #sample = np.sort(np.random.choice(flight.traj.shape[1],6000,replace=False))

    
    #_,idx1,_ = np.intersect1d(np.round(raw_traj[0]),spline_traj[0],assume_unique=True,return_indices=True)
    #flight.traj = raw_traj[:,idx1]
    
    #flight.traj = raw_traj[:,1800:9445]
    flight.traj = raw_traj[:,1846:]
    #flight.traj = raw_traj[:,2060:]

    #flight.tracks = flight.tracks[:,sample]

    flight.set_visibility(final=final)
    
    beta = flight.beta[0]
    alpha = flight.alpha[0]
    #rs = flight.rs[0]
    rs = [flight.cameras[i].rs for i in range(len(beta))]
    
    Reprojection_Error = {}
    for i in motion_weights: 

        res_mot, model_mot = common.optimize_all(flight.cameras,Tracks,flight.traj,flight.visible,spline=spline,
                                include_K=include_K,max_iter=max_iter,motion=motion,motion_weights=i,
                                prior=prior,distortion=include_d,beta=beta,alpha=alpha,rs=rs,final=True,opt_beta=opt_beta,
                                calibration=opt_calibration,verbose=2,sparse=True,rs_cor=include_rs,rs_bounds=rsbounds)

        print('\nTime: {}\n'.format(datetime.now()-start))

        # After BA: interpret results
        if opt_beta:
                #if include_alpha:
                #    flight.beta[0], flight.alpha[0], flight.cameras, flight.traj = model_mot[0], model_mot[1], model_mot[2], model_mot[3]
            if include_rs:
                if opt_calibration:
                    flight.beta[0], flight.rs[0], flight.cameras, flight.traj = model_mot[0], model_mot[1], model_mot[2],model_mot[3]
                else:
                    flight.beta[0], flight.rs[0],flight.traj = model_mot[0], model_mot[1], model_mot[2]

            else:
                if opt_calibration:
                    flight.beta[0], flight.cameras,flight.traj = model_mot[0], model_mot[1], model_mot[2]
                else:
                    flight.beta[0], flight.traj = model_mot[0], model_mot[1]
            
        else:
            if include_rs:
                if opt_calibration:
                    flight.rs[0],flight.cameras, flight.traj = model_mot[0], model_mot[1], model_mot[2]
                else:
                    flight.rs[0], flight.traj = model_mot[0], model_mot[1]
            else:
                if opt_calibration:
                    flight.cameras, flight.traj = model_mot[0], model_mot[1]
                else:
                    flight.traj = model_mot[0]

        #else:
        #    flight.traj = model_mot[0]

        # Check Result    
        # M = transformations.affine_matrix_from_points(flight.traj[1:],traj_2n,shear=False,scale=True)
        # traj_4 = np.dot(M,util.homogeneous(flight.traj[1:]))
        # traj_4 /= traj_4[-1]
        # vis.show_tf_trajectory_3D(traj_4[:3],X_dst=traj_2n)
        
        #flight.undistort_detections(apply=True)
        #vis.show_trajectory_3D(spline_traj[1:],flight.traj[1:],line=False,title='Raw Reconstruction vs GPS (1st flight)',pause=False)
         
        #vis.show_trajectory_3D(spline_traj[1:],flight.traj[1:],line=False,title='Raw Reconstruction vs GPS (1st flight)',pause=False)
        
        flight.set_tracks(auto=True,raw=raw,rs_cor=True)
        # Check reprojection error
        print('\nAfter optimizing Raw detections, beta:{}, d:{},spline:{},motion:{}'.format(include_b,include_d,use_spline,motion))
        Reprojection_Error[i] = [
            flight.error_cam(f1,final=True),
            flight.error_cam(f2,final=True),
            flight.error_cam(f3,final=True),
            flight.error_cam(f4,final=True)
            ]
        
       # flight.set_tracks(raw=raw, alpha=flight.alpha[0],final=True,traj_scale=30)

        #int_trajs = common.interpolate_track(flight.tracks,spline_traj)

        #final_raw_traj = common.combine_traj(int_trajs,[f1,f2,f3,f4])
        #flight.traj = final_raw_traj

        # Reprojection_Error[i] = [
        #     flight.error_cam(f1),
        #     flight.error_cam(f2),
        #     flight.error_cam(f3),
        #     flight.error_cam(f4)
        #     ]

        
        if motion:
            sigma = 'points_{}_wght_{}'.format(prior,np.format_float_scientific(i,3))
        else:
            sigma = 'points_no_MP'
        with open('data/GPS_to_reconstruction/reconstruction_cam4_corrected/RS_Test/flight_1_'+str(sigma)+'.pkl','wb') as f:
            pickle.dump(flight, f)

        #with open('data/GPS_to_reconstruction/MP_norm_pnts/points/errors/flight_1_rpjct_err_'+str(sigma)+'.pkl','wb') as f:
        #    pickle.dump(Reprojection_Error, f)

    print('Finished')
