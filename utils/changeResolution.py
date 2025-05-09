
d = np.load('voidSeeded_3.000pc_potts-12_3d.975.npy')
# (120, 20, 200)
# dc = np.resize(d, (100, 10, 60)).T
from scipy.ndimage import zoom
dc = zoom(d, (0.5, 0.5, 0.5), order=0)
np.save('dc.npy', dc)
