
import numpy as np

tableFile = 'MgRve_8x8x8.txt'
d = np.loadtxt(tableFile, skiprows=4)
d[:,[0,1,2]] /= 64
d[:,[0,1,2]] -= 1
d[:,[0,1,2]] /= 2

d = np.reshape(d, [8, 8, 8])
