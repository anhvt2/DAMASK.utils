
import numpy as np
import os, glob
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.figure()

dimCellList = [2, 4, 8, 16, 32]
for i in range(len(dimCellList)):
	dimCell = dimCellList[i]
	num_obs = len(glob.glob('%sx%sx%s*output.dat' % (dimCell, dimCell, dimCell)))
	print(f"Found {num_obs} observations at {dimCell}x{dimCell}x{dimCell}")

	for fileName in glob.glob('%sx%sx%s*output.dat' % (dimCell, dimCell, dimCell)):
		o = np.loadtxt(fileName)
		strain = o[:,0]
		stress = o[:,1]
		strain = np.insert(strain, 0, 0)
		stress = np.insert(stress, 0, 0)

		#
		if np.any(stress > 5e2) or np.any(np.diff(stress) < 0):
			print(f"{fileName} is not usable.")
		else:
			interp_strain = np.linspace(0, 0.1, num=201)
			splineInterp = interp1d(strain, stress, kind='cubic', fill_value='extrapolate')
			interp_stress = splineInterp(interp_strain)
			plt.plot(interp_strain, interp_stress)

plt.show()

