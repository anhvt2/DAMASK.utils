import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

phi = np.linspace(1/2*np.pi,np.pi, 256).reshape(256, 1) # the angle of the projection in the xy-plane
theta = np.linspace(0.5*np.pi, np.pi, 256).reshape(-1, 256) # the angle from the polar axis, ie the polar angle
radius = 4

# Transformation formulae for a spherical coordinate system.
ratio=3
x = ratio*radius*np.sin(theta)*np.cos(phi)
y = radius*np.sin(theta)*np.sin(phi)
z = radius*np.cos(theta)

fig = plt.figure()  # Square figure
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b')
# ax.set_box_aspect([ratio,1,1])
plt.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
