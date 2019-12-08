import numpy as np
import pickle
from tools import visualization as vis
from datetime import datetime
from reconstruction import common

# Initialize a scene from the json template
flight = common.create_scene('config_example.json')

# Truncate detections
flight.cut_detection(second=flight.settings['cut_detection_second'])

# Add prior alpha
flight.init_alpha()

# Convert raw detections into the global timeline
flight.detection_to_global()

# Initialize the first 3D trajectory
flight.init_traj(error=flight.settings['thres_Fmatix'])

# Convert discrete trajectory to spline representation
flight.traj_to_spline(smooth_factor=flight.settings['smooth_factor'])


'''---------------Incremental reconstruction----------------'''
start = datetime.now()
np.set_printoptions(precision=4)

cam_temp = 2

while True:
    print('\n----------------- Bundle Adjustment with {} cameras -----------------'.format(cam_temp))
    print('\nMean error of each camera before BA:   ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

    # Bundle adjustment
    res = flight.BA_mot(cam_temp, rs=flight.settings['rolling_shutter'],motion=True,motion_weights=0)
    #res = flight.BA(cam_temp, rs=flight.settings['rolling_shutter'])

    print('\nMean error of each camera after BA:    ', np.asarray([np.mean(flight.error_cam(x,motion=True,mode='dist',norm=False)) for x in flight.sequence[:cam_temp]]))

    flight.remove_outliers(flight.sequence[:cam_temp],thres=flight.settings['thres_outlier'])

    res = flight.BA_mot(cam_temp, rs=flight.settings['rolling_shutter'],motion=True,motion_weights=0)

    print('\nMean error of each camera after second BA:    ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

    #Compute reconstruction with motion prior
    #res = flight.BA_mot(cam_temp,rs=flight.settings['rolling_shutter'],motion=True,motion_weights=1)

    #print('\nMean error of each camera after second BA with MP reg.: ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

    num_end = flight.numCam if flight.find_order else len(flight.sequence)
    if cam_temp == num_end:
        print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
        break
    
    # Select the next camera if not pre-defined
    flight.select_most_overlap()

    # Add the next camera and get its pose
    flight.get_camera_pose(flight.sequence[cam_temp])

    # Triangulate new points and update the 3D spline
    flight.triangulate(flight.sequence[cam_temp], flight.sequence[:cam_temp], thres=flight.settings['thres_triangulation'],
                       factor_t2s=flight.settings['smooth_factor'], factor_s2t=flight.settings['sampling_rate'])

    print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
    cam_temp += 1

# Visualize the 3D trajectory
flight.spline_to_traj(sampling_rate=1)
vis.show_trajectory_3D(flight.traj[1:],line=False)

with open(flight.settings['path_output'],'wb') as f:
    pickle.dump(flight, f)


print('Finish!')