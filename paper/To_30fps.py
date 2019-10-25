import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt


fps_ori = 49.508892
fps_tar = 30
det_path = './data/paper/crash/detection/out_xiaomi_1.txt'
det_in = np.loadtxt(det_path).T

idx = det_in[2]/fps_ori*fps_tar
idx_int = np.arange(np.ceil(idx[0]),np.floor(idx[-1]))
idx_final = np.array([0])
for i in idx_int:
    if np.in1d(int(i*fps_ori/fps_tar),det_in[2])*np.in1d(int(i*fps_ori/fps_tar)-1,det_in[2])*np.in1d(int(i*fps_ori/fps_tar)+1,det_in[2]):
        idx_final = np.append(idx_final,i)
idx_final = idx_final[1:]

tck,u = interpolate.splprep(det_in[:2],u=idx,s=0,k=3)
a = interpolate.splev(idx_final,tck)
det_out = np.array([a[0],a[1],idx_final])

plt.subplot(1,2,1)
plt.scatter(det_in[0],det_in[1])
plt.subplot(1,2,2)
plt.scatter(det_out[0],det_out[1])
plt.show()

plt.figure()
plt.scatter(det_in[0],det_in[1],s=25)
plt.scatter(det_out[0],det_out[1],s=15)
plt.show()

print('ratio of number of detections: {:.3f}'.format(det_out.shape[1] / det_in.shape[1]))
print('ratio of fps: {:.3f}'.format(fps_tar / fps_ori))

np.savetxt(det_path[:-4]+'_30'+det_path[-4:],det_out.T)

# for i in range(1,det_out.shape[1]-1):
#     if int(det_out[2,i]-det_out[2,i-1])!=1 and int(det_out[2,i+1]-det_out[2,i])!=1:
#         print('problem')

# det_ver = np.loadtxt('./data/detections_30/aaa_sonyg.txt').T
# plt.figure()
# plt.scatter(det_in[0],det_in[1],s=25)
# plt.scatter(det_ver[0],det_ver[1],s=15)
# plt.show()


print('\nFinish')