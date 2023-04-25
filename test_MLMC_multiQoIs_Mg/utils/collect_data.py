

""" 
	replace collect_data.sh
"""

import os, sys, datetime
import numpy as np

os.system('find . -name %s' % 'postProc > folders.list') 

folders_list = np.loadtxt('folders.list', dtype=str)
parentPath = os.getcwd()

os.system('rm -v {folderName,mesh_time_stamp,mesh_size,validity}.log')

def checkValidity(parentPath, folderName):
	# check validity
	logFile = open(parentPath + '/' + folderName + '/' + '../log.MultilevelEstimators-multiQoIs')
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
	# print(d)
	num_rows = d.shape[0]
	levels = d[:, 0]
	"""
		pass only two cases: 
		(1) IF level == 0 AND no NaN THEN pass
		(2) IF level > 0 AND levels are valid AND no NaN THEN pass
	"""
	validity = 0 # initialize
	if num_rows == 1 and levels == 0 and (not np.any(np.isnan(d[:, 1:]))):
		validity = 1
	if num_rows == 2 and levels[0] - levels[1] == 1 and (not np.any(np.isnan(d[:, 1:]))):
		validity = 1
	return validity

for folderStr in folders_list:
	os.chdir(parentPath + '/' + folderStr)
	folderName = folderStr.split('/')[1]
	mesh_time_stamp = folderStr.split('/')[3]
	mesh_size = mesh_time_stamp.split('-')[0].split('x')[0]
	time_stamp = mesh_time_stamp.split('-')[1:]
	# write metadata
	f = open(parentPath + '/' + 'folderName.log', 'a+') # e.g. hpc_level-0_sample-4747
	f.write('%s\n' % folderName)
	f.close()
	f = open(parentPath + '/' + 'mesh_time_stamp.log', 'a+') # e.g. '2x2x2-23-04-19-15-44-03'
	f.write('%s\n' % mesh_time_stamp)
	f.close()
	f = open(parentPath + '/' + 'mesh_size.log', 'a+') # e.g. '2'
	f.write('%s\n' % mesh_size)
	f.close()
	f = open(parentPath + '/' + 'time_stamp.log', 'a+') # e.g. hpc_level-0_sample-4747
	f.write('%s\n' % time_stamp)
	f.close()
	# calculate elapsed time
	validity = checkValidity(parentPath, folderName)
	f = open(parentPath + '/' + 'validity.log', 'a+')
	f.write('%s\n' % validity)
	f.close()
	print(f"done {folderName}")
	# copy data






