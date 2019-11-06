import numpy as np
from scipy.interpolate import splprep, splev
from matplotlib import pyplot as plt

# Create two curves
c1 = np.array([[1,2,3,4,5],[5,4,3,2,1]], dtype=float)
c2 = np.array([[21,22,23,24,25],[12,11.5,11,10.5,10]], dtype=float)

# Add some noise
c1[1] = c1[1] + np.random.randn(5)*0.1
c2[1] = c2[1] + np.random.randn(5)*0.1

# Option 1: fit One spline
c = np.hstack((c1,c2))
tck,u = splprep(c,u=c[0])

# Option 2: fit Two splines
tck1,u1 = splprep(c1,u=c1[0])
tck2,u2 = splprep(c2,u=c2[0])

# Option 1: Interpolation from 1 curve
x = np.concatenate((np.arange(1,5,0.1), np.arange(21,25,0.1)))
y = np.asarray(splev(x, tck))
[y_1,y_2] = np.split(y,2,axis=1)

m = np.arange(5,21,0.1)
n = np.asarray(splev(m, tck))

# Option 2: Interpolation from 2 curves
x1 = np.arange(1,5,0.1)
x2 = np.arange(21,25,0.1)
y1 = np.asarray(splev(x1, tck1))
y2 = np.asarray(splev(x2, tck2))

# Comparison
d1 = abs(y_1[1] - y1[1])
d2 = abs(y_2[1] - y2[1])
print('Sum of differences on interpolated points, {:.2f} for curve 1 and {:.2f} for curve 2'.format(sum(d1), sum(d2)))

# Visualize Option 1
plt.figure()
plt.subplot(1,2,1)
plt.plot(c1[0],c1[1],'r')
plt.plot(c2[0],c2[1],'r')
plt.plot(y_1[0],y_1[1],'b')
plt.plot(y_2[0],y_2[1],'b')
plt.plot(n[0],n[1],'g')

# Visualize Option 2
plt.subplot(1,2,2)
plt.plot(c1[0],c1[1],'r')
plt.plot(c2[0],c2[1],'r')
plt.plot(y1[0],y1[1],'b')
plt.plot(y2[0],y2[1],'b')


plt.show()

print('finish!')