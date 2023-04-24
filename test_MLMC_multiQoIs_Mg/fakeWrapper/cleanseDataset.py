import numpy as np
import matplotlib.pyplot as plt
import os

## deprecated - replaced by import_dataset()
# os.system('cat ../log.MultilevelEstimators-multiQoIs.1 >>  log.MultilevelEstimators-multiQoIs')
# os.system('cat ../log.MultilevelEstimators-multiQoIs.2 >> log.MultilevelEstimators-multiQoIs')
# os.system('cat ../hpc-run-1.log.MultilevelEstimators-multiQoIs >> log.MultilevelEstimators-multiQoIs')
# os.system('cat ../hpc-run-2.log.MultilevelEstimators-multiQoIs >> log.MultilevelEstimators-multiQoIs')
# os.system('cat ../hpc-run-3.log.MultilevelEstimators-multiQoIs >> log.MultilevelEstimators-multiQoIs')

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
	d = []
	for i in range(len(txt)):
		txt[i] = txt[i].replace('Collocated von Mises stresses at ', '')
		txt[i] = txt[i].replace(' is ', ',') # replace with a comma
		txt[i] = txt[i].replace('\n', '') 
		tmp_list = txt[i].split(',')
		d += [tmp_list]

	d = np.array(d, dtype=float)
	# check
	tossaway_idx = []
	# print(d.shape[0]) # debug
	ii = 0 # debug
	while ii < d.shape[0] - 1:
		# print(ii)
		if d[ii,0] - d[ii+1,0] == 1:
			# print(f"Index {ii} valid in {datasetFileName}. Skip index {ii+1}.") # debug
			ii += 1 # skip next index if valid
		# elif d[ii,0] == 0: # debug
		# 	print(f"Index {ii} valid in {datasetFileName}.") # debug
		else:
			if d[ii,0] != 0 and d[ii,0] - d[ii+1,0] != 1:
				# print(f"Index {ii} is not valid in {datasetFileName}: lonely sample. -- Discard.") # debug
				# print(f"{d[ii]}, {d[ii+1]}")
				# print(txt[ii])
				tossaway_idx.append(ii)
			## deprecated: move to later
			# if np.any(np.isnan(d[ii,:])):
			# 	if d[ii,0] == 0:
			# 		print(f"Index {ii} is not valid in {datasetFileName}: NaN found. -- Discard.")
			# 		tossaway_idx.append(ii)
			# 		print(f"{ii}: {np.any(np.isnan(d[ii,:]))}") # debug
			# 	else:
			# 		tossaway_idx.append(ii)
			# 		print(f"Index {ii} is not valid in {datasetFileName}: NaN found. Discard.")
			# 		if d[ii,0] - d[ii+1,0] == 1:
			# 			tossaway_idx.append(ii+1)
			# 			print(f"Index {ii+1} is not valid in {datasetFileName}: NaN found in previous level. -- Discard.")
			# 			ii += 1
		ii += 1

	tossaway_idx = np.sort(np.unique(np.array(tossaway_idx)))

	# print(len(txt))
	for i in range(len(tossaway_idx) - 1, -1, -1):
		txt.pop(tossaway_idx[i])
	# import
	logFile = open('log.MultilevelEstimators-multiQoIs', 'a+')
	for i in range(len(txt)):
		logFile.write(txt[i])
		logFile.write('\n')
	logFile.close()

import_dataset('../log.MultilevelEstimators-multiQoIs.1')
import_dataset('../log.MultilevelEstimators-multiQoIs.2')

for i in range(1,5):
	import_dataset('../hpc-run-%d.log.MultilevelEstimators-multiQoIs' % i)

# import_dataset('../hpc-run-5.log.MultilevelEstimators-multiQoIs') # nan - debug

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

for i in range(d.shape[0]):
	# removal condition: nan
	if np.any(np.isnan(d[i, 1:])):
		del_idx.append(i)

del_idx = np.unique(np.sort(np.array(del_idx)))
print(del_idx)

# diagnostics
print("Pre-cleanse Statistics")
print(f"Start with {d.shape[0]} samples.")
print(f"Remove {len(del_idx)} samples.")

del_level = d[del_idx, 0]

for level in range(num_levels):
	print(f"Remove {np.sum(d[del_idx, 0] == level)} samples at level {level}.")

d = np.delete(d, del_idx, axis=0)
print(f"End with {d.shape[0]} samples.")
print('\n')
print("Post-cleanse Statistics")
for level in range(num_levels):
	print(f"Found {np.sum(d[:, 0] == level)} samples at level {level}.")

# save dataset
np.savetxt("MultilevelEstimators-multiQoIs.dat", d, delimiter=",", header="level, q0, q1, q2, q3, q4, q5, q6, q7, q8, q9", fmt="%d, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e")

