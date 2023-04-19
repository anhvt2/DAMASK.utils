
import numpy as np
import argparse
import os, sys, glob, datetime

cost_per_level = [39,    365,    1955,    3305,    12487] # taken from run_multilevel_multiple_qoi.py
cost_per_level = np.array(cost_per_level)
num_level = len(cost_per_level)
currentPath = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("-ms", "--max_sample", type=int) # desired number of samples at the highest level of fidelity



args = parser.parse_args()
max_sample = int(args.max_sample)

num_samples = np.array(max_sample * np.max(cost_per_level) / cost_per_level, dtype=int)

for i in range(num_level):
# 	print(f"Need {int( max_sample * cost_per_level[-1] / cost_per_level[i])} samples at level {i}")
	print(f"Need {num_samples[i]} samples at level {i}")
	num_sample = num_samples[i]

	for j in range(num_sample):
		folderName = "hpc_level-%d_sample-%d" % (i, j)
		os.system('cp -r template/ %s' % folderName)
		print(f"Create {folderName}.")
		os.chdir(currentPath + '/' + folderName)

		# read
		slurmfile = open('sbatch.damask.srn')
		slurmtext = slurmfile.readlines()
		slurmfile.close()
		# modify query (level) based on char location
		old_string = slurmtext[44]
		list_str = list(old_string)
		list_str[46] = str(i)
		new_string = ''.join(list_str)
		slurmtext[44] = new_string
		# write
		slurmfile = open('sbatch.damask.srn', 'w') # can be 'r', 'w', 'a', 'r+'
		for lineNo in range(len(slurmtext)):
			slurmfile.write(slurmtext[lineNo])
		slurmfile.close()
		os.chdir(currentPath)
	os.chdir(currentPath)

