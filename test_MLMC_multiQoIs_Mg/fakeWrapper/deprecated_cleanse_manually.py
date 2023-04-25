
import numpy as np

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
		if d[ii,0] != 0 and d[ii,0] - d[ii+1,0] != 1:
			# print(f"Index {ii} is not valid in {datasetFileName}: lonely sample. -- Discard.") # debug
			# print(f"{d[ii]}, {d[ii+1]}")
			# print(txt[ii])
			tossaway_idx.append(ii)
		if np.any(np.abs(d[ii, 1:]) > 5e2):
			print(f"{ii}: {txt[ii]}")
			if d[ii,0] == 0:
				tossaway_idx.append(ii)
			else:
				if d[ii,0] - d[ii+1,0] == 1:
					tossaway_idx.append(ii+1)
				# 	ii += 1
				# elif d[ii-1,0] - d[ii,0] == 1:
				# 	tossaway_idx.append(ii-1)

		ii += 1

	tossaway_idx = np.sort(np.unique(np.array(tossaway_idx)))

	# print(len(txt))
	for i in range(len(tossaway_idx) - 1, -1, -1):
		txt.pop(tossaway_idx[i])
	
	# import
	new_file_name = datasetFileName.split('/')[-1]
	new_file_handler = 'cleansed.' + new_file_name
	logFile = open(new_file_handler, 'a+')
	for i in range(len(txt)):
		# logFile.write(txt[i])
		# print(len(txt[i]))
		# print(txt[i][2:])
		logFile.write('Collocated von Mises stresses at %s is %s' % (txt[i][0], txt[i][2:]))
		logFile.write('\n')
	logFile.close()

import_dataset('cleansed.log.MultilevelEstimators-multiQoIs.2')
