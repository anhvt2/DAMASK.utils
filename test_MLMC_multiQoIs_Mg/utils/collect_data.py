

""" 
	replace collect_data.sh
"""

import os, sys, datetime
import numpy as np

os.system('find . -name %s' % 'postProc > folders.list') 

folders_list = np.loadtxt('folders.list', dtype=str)
parentPath = os.getcwd()

os.system('rm -v folderName.log mesh_date_stamp.log validity.log')

def checkValidity(parentPath, folderName)
	# check validity
	logFile = open(parentPath + '/' + folderStr + '/' + '../log.MultilevelEstimators-multiQoIs')
	txt = logFile.readlines()
	logFile.close()
	d = []
	for i in range(len(txt)):
		txt[i] = txt[i].replace('Collocated von Mises stresses at ', '')
		txt[i] = txt[i].replace(' is ', ',') # replace with a comma
		txt[i] = txt[i].replace('\n', '') 
		tmp_list = txt[i].split(',')
		d += [tmp_list]
	d = np.array(d)
	print(d)
	num_rows = d.shape[0]
	validity = 0
	if num_rows == 1 and d[0,0] == 1:
		validity = 1
	return validity

for folderStr in folders_list:
	os.chdir(parentPath + '/' + folderStr)
	folderName = folderStr.split('/')[1]
	mesh_date_stamp = folderStr.split('/')[3]
	# write metadata
	f = open('folderName.log', 'a+')
	f.write('%s\n' % folderName)
	f.close()
	f = open('mesh_date_stamp.log', 'a+')
	f.write('%s\n' % mesh_date_stamp)
	f.close()
	validity = checkValidity(parentPath, folderName)
	f = open('validity.log', 'a+')
	f.write('%s\n' % valid)
	f.close()


	# copy data

	# calculate elapsed time



