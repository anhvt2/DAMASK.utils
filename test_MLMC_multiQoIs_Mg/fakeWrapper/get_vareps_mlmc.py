
import numpy as np
vareps_ub = 3.51e-1 # upper bound - determined by warm-up MLMC limit
vareps_lb = 5.0e-3 # lower bound

f = open('vareps_mlmc.dat', 'w')
for log_vareps in np.linspace(np.log10(vareps_ub), np.log10(vareps_lb), num=100):
	f.write('%.8e' % log_vareps)

f.close()

