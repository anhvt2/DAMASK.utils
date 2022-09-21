import numpy as np
i = np.loadtxt('input.dat')
np.savetxt('input.dat', [i], delimiter=',', fmt='%.18e')
