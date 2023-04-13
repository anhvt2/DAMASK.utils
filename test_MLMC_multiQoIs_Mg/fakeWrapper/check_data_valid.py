
import numpy as np

data = np.loadtxt("MultilevelEstimators-multiQoIs.dat", delimiter=",")
d = data[:,0]

for i in range(len(d) - 1):
	if not (d[i] - d[i+1] == 1 or d[i-1] - d[i] == 1) and d[i] != 0:
		print(f"Found index {i} with level {d[i]}")
