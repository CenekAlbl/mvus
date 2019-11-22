# This script trys to solve the BA bug for the 6th camera

import numpy as np
import pickle
import tools.visualization as vis
from datetime import datetime


data_file = './data/paper/MS/trajectory/flight_temp1.pkl'
with open(data_file, 'rb') as file:
    flight = pickle.load(file)

for i in range(flight.numCam):
    cam_id = flight.sequence[i]
    if flight.cameras[cam_id].P is not None:
        continue
    else:
        cam_temp = i
        break

start = datetime.now()
np.set_printoptions(precision=4)

flight.sequence = [2,4,5,3,5,0]

while True:
    # get camera pose
    flight.get_camera_pose(flight.sequence[cam_temp], error = 30)

    # Triangulate
    flight.triangulate(flight.sequence[cam_temp], flight.sequence[:cam_temp], thres=flight.setting['tri_thres'], factor_t2s=flight.setting['smooth'], factor_s2t=flight.setting['sampling'])

    cam_temp += 1

    print('\n----------------- Bundle Adjustment with {} cameras -----------------'.format(cam_temp))
    print('\nMean error of each camera before BA:   ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

    # Bundle adjustment
    res = flight.BA(cam_temp)

    print('\nMean error of each camera after BA:    ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

    flight.remove_outliers(flight.sequence[:cam_temp],thres=flight.setting['outlier_thres'])

    res = flight.BA(cam_temp)

    print('\nMean error of each camera after second BA:    ', np.asarray([np.mean(flight.error_cam(x)) for x in flight.sequence[:cam_temp]]))

    flight.spline_to_traj()
    vis.show_trajectory_3D(flight.traj[1:])

    # with open('./data/paper/MS/trajectory/flight_temp.pkl','wb') as f:
    #     pickle.dump(flight, f)

    if cam_temp == len(flight.sequence):
        print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))
        break

    print('\nTotal time: {}\n\n\n'.format(datetime.now()-start))


print('Finish!')