#!/usr/bin/env python3
import numpy as np
from scipy.ndimage import zoom

D = np.load('voidSeeded_3.000pc_potts-12_3d.975.npy')
DC = zoom(D, (0.5, 0.5, 0.5), order=0)
np.save('dc.npy', DC)
