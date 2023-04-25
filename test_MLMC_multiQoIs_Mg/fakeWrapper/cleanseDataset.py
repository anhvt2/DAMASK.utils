import numpy as np
import matplotlib.pyplot as plt
import os

os.system('rm -v log.MultilevelEstimators-multiQoIs')

def import_dataset(datasetFileName):
	"""
	Usage:
		check and import dataset to log.MultilevelEstimators-multiQoIs
	"""
	# read
	logFile = open(datasetFileName)
	txt = logFile.readlines()
	logFile.close()
	# import
	logFile = open('log.MultilevelEstimators-multiQoIs', 'a+')
	for i in range(len(txt)):
		logFile.write(txt[i])
	logFile.close()
	# print
	print(f"Import {datasetFileName} with {len(txt)} samples.")

# import
# import_dataset('../log.MultilevelEstimators-multiQoIs.1')
import_dataset('cleansed.log.MultilevelEstimators-multiQoIs.2')

for i in range(1,6):
	import_dataset('../hpc-run-%d.log.MultilevelEstimators-multiQoIs' % i)

# read dataset
logFile = open('log.MultilevelEstimators-multiQoIs')
txt = logFile.readlines()
logFile.close()

d = []

# cleanse dataset
for i in range(len(txt)):
	txt[i] = txt[i].replace('Collocated von Mises stresses at ', '')
	txt[i] = txt[i].replace(' is ', ',') # replace with a comma
	txt[i] = txt[i].replace('\n', '') 
	tmp_list = txt[i].split(',')
	d += [tmp_list]

d = np.array(d, dtype=float)
num_levels = int(np.max(d[:,0]) - np.min(d[:,0]) + 1)

# diagnostics
print("Post-cleanse Statistics")
for level in range(num_levels):
	print(f"Found {np.sum(d[:, 0] == level)} samples at level {level}.")

# save dataset
np.savetxt("MultilevelEstimators-multiQoIs.dat", d, delimiter=",", header="level, q0, q1, q2, q3, q4, q5, q6, q7, q8, q9", fmt="%d, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e")
# np.savetxt("MultilevelEstimators-multiQoIs.dat", d, delimiter=",", header="level, q0, q1, q2, q3, q4, q5, q6, q7, q8, q9", fmt="%d, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f")
