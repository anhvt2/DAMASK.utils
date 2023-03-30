
import numpy as np
import matplotlib.pyplot as plt



logFile = open('log.MultilevelEstimators-multiQoIs')
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

levels = np.unique(d[:,0]) # get array of levels
d0 = d[ np.where(d[:,0] == 0) ] # level 0 only: could be generalized -- brute force

sample_incre = 20 # sample increment
num_samples = np.arange(1, int(np.floor(d0.shape[0] / sample_incre)) + 1) * sample_incre

plt.figure()
for num_sample in num_samples:
	# randomly select a subset and sort, and repeat for a Monte Carlo times
	tmp_var = []
	for j in range(1000):
		idx = sorted(np.random.choice(d0.shape[0], size=num_sample, replace=False))
		sub_d0 = d0[idx]
		var_multi_qoi = np.var(sub_d0, axis=0)
		tmp_var += [var_multi_qoi[-1]]
	plt.plot(num_sample, np.mean(tmp_var), 'bo', ms=5)

plt.xlabel('samples', fontsize=24)
plt.ylabel('variance', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tick_params(axis='both', which='minor', labelsize=24)
plt.xscale('log',base=10) 
plt.show()
