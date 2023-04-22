
import numpy as np
import argparse
import os, sys, glob, datetime

cost_per_level = [39,    365,    1955,    3305,    12487] # taken from run_multilevel_multiple_qoi.py
cost_per_level = np.array(cost_per_level)
num_level = len(cost_per_level)
currentPath = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("-ms", "--samples_max_level", type=int) # desired number of samples at the highest level of fidelity
parser.add_argument("-min_level", "--min_level", type=int, default=0) # minimum starting level, default: 0
parser.add_argument("-sbatch", "--slurm_file_name", type=str, default='sbatch.damask.srn') # slurm file name, default: on SRN

args = parser.parse_args()
samples_max_level = int(args.samples_max_level)
min_level = int(args.min_level)
slurm_file_name = args.slurm_file_name

num_samples = np.array(samples_max_level * np.max(cost_per_level) / cost_per_level, dtype=int)

### remove all related folders
os.system('scancel -u anhtran')
os.system('rm -rfv hpc_level-*')

for i in range(min_level, num_level):
# 	print(f"Need {int( samples_max_level * cost_per_level[-1] / cost_per_level[i])} samples at level {i}")
	print(f"Need {num_samples[i]} samples at level {i}")
	num_sample = num_samples[i]
	cost = cost_per_level[i]
	# convert seconds to seconds/minutes/hours
	# https://stackoverflow.com/questions/775049/how-do-i-convert-seconds-to-hours-minutes-and-seconds
	minutes_cost, seconds_cost = divmod(cost, 60)
	hours_cost, minutes_cost = divmod(minutes_cost, 60)

	for j in range(num_sample):
		folderName = "hpc_level-%d_sample-%d" % (i, j)
		os.system('cp -r template/ %s' % folderName)
		print(f"Create folder {folderName}")
		os.chdir(currentPath + '/' + folderName)

		# read
		slurmfile = open(slurm_file_name)
		slurmtext = slurmfile.readlines()
		slurmfile.close()
		
		# modify query (level) based on char location
		old_string = slurmtext[44]
		list_str = list(old_string)
		list_str[51] = str(i)
		new_string = ''.join(list_str)
		slurmtext[44] = new_string

		# add to 'short' queue to promote shorter wait time
		if hours_cost < 4:
			slurmtext[3] = '#SBATCH --time=04:00:00               # Wall clock time (HH:MM:SS) - once the job exceeds this time, the job will be terminated (default is 5 minutes)\n'
			slurmtext[6] = '#SBATCH --partition=short,batch       # partition/queue name: short or batch\n'
		
		# write
		slurmfile = open(slurm_file_name, 'w') # can be 'r', 'w', 'a', 'r+'
		for lineNo in range(len(slurmtext)):
			slurmfile.write(slurmtext[lineNo])
		slurmfile.close()
		
		os.chdir(currentPath)
	os.chdir(currentPath)

# ## automatic submission -- deprecated: replaced by autoSubmit.sh
# for i in range(num_level):
# 	num_sample = num_samples[i]
# 	for j in range(num_sample):
# 		folderName = "hpc_level-%d_sample-%d" % (i, j)
# 		print(f"Submit Slurm job in {folderName}")
# 		os.chdir(currentPath + '/' + folderName)
# 		os.system('ssubmit')
# 	os.chdir(currentPath)
# os.chdir(currentPath)
