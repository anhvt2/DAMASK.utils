
import numpy as np
import os, glob
import argparse
import time
import datetime
import socket

### adopt from wrapper_multilevel_multiple_qoi.py
parser = argparse.ArgumentParser()
parser.add_argument("-level", "--level", type=int)
parser.add_argument("-nb_of_qoi", "--nb_of_qoi", type=int)

args = parser.parse_args()
level = int(args.level);
nb_of_qoi = int(args.nb_of_qoi) # currently not being used

### load dataset
d = np.loadtxt('MultilevelEstimators-multiQoIs.dat', skiprows=1, delimiter=',')
levels = d[:,0]

### lookup data and return only FIRST correct result: query_index
possible_idx = np.where(levels == level)[0]

# remove last index in case level = 0 to avoid index overflow
if level == 0:
	possible_idx = np.delete(possible_idx, len(possible_idx) - 1)

### search and return query_index
for i in possible_idx:
	if i < d.shape[0] - 1:
		if level == 0:
			query_index = possible_idx[0]
			break
		else:
			if levels[i] - levels[i+1] == 1: # check if level decreasement is satisfied
				query_index = i
				# # verbose
				# print(f"levels[{i}] = {int(levels[i])}")
				# print(f"levels[{i+1}] = {int(levels[i+1])}")
				# print(f"Found query_index = {i}")
				break

if level == 0:
	results = d[ query_index ]
	stress_only_results = results[1:]
else:
	results = d[ [query_index, query_index + 1] ] # return level and (level - 1)
	stress_only_results = results[:,1:]

### print results w/ identical interface
if level == 0:
	vmStress2str = np.array(stress_only_results, dtype=str)
	str2print = ', '.join(vmStress2str) # construct a string to print on screen
	print("Collocated von Mises stresses at %d is %s " % (level, str2print))
else:
	# (level)
	vmStress2str_current_level = np.array(stress_only_results[0], dtype=str)
	str2print_current_level = ', '.join(vmStress2str_current_level)
	# (level - 1)
	vmStress2str_below_level = np.array(stress_only_results[1], dtype=str)
	str2print_below_level = ', '.join(vmStress2str_below_level) # construct a string to print on screen
	print("Collocated von Mises stresses at %d is %s " % (level, str2print_current_level))
	print("Collocated von Mises stresses at %d is %s " % (level - 1, str2print_below_level))

### reduce results
if level == 0:
	reduced_d = np.delete(d, (query_index), axis=0) # remove the corresponding results after query
else:
	reduced_d = np.delete(d, (query_index, query_index + 1), axis=0) # remove the corresponding results after query

### save reduced dataset after eliminating appropriate rows
np.savetxt("MultilevelEstimators-multiQoIs.dat", reduced_d, delimiter=",", header="level, q0, q1, q2, q3, q4, q5, q6, q7, q8, q9", fmt="%d, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e, %.8e")



