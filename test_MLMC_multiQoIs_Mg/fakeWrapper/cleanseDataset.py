import numpy as np
import matplotlib.pyplot as plt
import os

os.system('cat ../log.MultilevelEstimators-multiQoIs.1 >  log.MultilevelEstimators-multiQoIs')
os.system('cat ../log.MultilevelEstimators-multiQoIs.2 >> log.MultilevelEstimators-multiQoIs')
os.system('cat ../log.MultilevelEstimators-multiQoIs.3 >> log.MultilevelEstimators-multiQoIs')

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

del_idx = []
for i in range(d.shape[0]):
	# removal condition: out of range stress > threshold
	if np.any(d[i,:] > 5e2):
		# print(i)
		del_idx.append(i)
		if d[i, 0] - d[i+1, 0] == 1:
			# print(i+1)
			del_idx.append(i+1)
		elif d[i-1, 0] - d[i, 0] == 1:
			# print(i-1)
			del_idx.append(i-1)
	# removal condition: non-monotonic stress
	if np.any(np.diff(d[i, 1:]) < 0):
		# print(i)
		del_idx.append(i)

# diagnostics
print(f"Start with {d.shape[0]} samples.")
print(f"Remove {len(del_idx)} samples.")
del_idx = np.array(del_idx)
del_level = d[del_idx, 0]

for level in range(num_levels):
	print(f"Specifically, remove {np.sum(d[del_idx, 0] == level)} samples at level {level}.")

d = np.delete(d, del_idx, axis=0)
print(f"End with {d.shape[0]} samples.")

# save dataset
np.savetxt("MultilevelEstimators-multiQoIs.dat", d, delimiter=",", header="level, q0, q1, q2, q3, q4, q5, q6, q7, q8, q9", fmt="%d, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e")

