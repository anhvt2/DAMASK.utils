
import numpy as np
import os
import time
import scipy
import numpy.linalg as nla


t_start = time.time()
d_MisesCauchy = np.load('d_MisesCauchy.npy')
print(f'Elapsed time: {time.time() - t_start:<.2f} seconds') 
# Elapsed time: 774.08 seconds

d_MisesLnV = np.load('d_MisesLnV.npy')
print(f'Elapsed time: {time.time() - t_start:<.2f} seconds')
# Elapsed time: 892.80 seconds

l_MisesCauchy = np.linalg.norm(d_MisesCauchy, axis=0)
nzMisesCauchy = np.count_nonzero(l_MisesCauchy) # nz: 10969; z: 290
d_MisesCauchy = d_MisesCauchy[:,:nzMisesCauchy]

l_MisesLnV = np.linalg.norm(d_MisesLnV, axis=0)
nzMisesLnV = np.count_nonzero(l_MisesLnV) # nz:; z:
d_MisesLnV = d_MisesLnV[:,:nzMisesLnV]

meanMisesCauchy = np.mean(d_MisesCauchy, axis=1)
meanMisesLnV = np.mean(d_MisesLnV, axis=1)

d_MisesCauchy = d_MisesCauchy - np.atleast_2d(meanMisesCauchy).T
d_MisesLnV = d_MisesLnV - np.atleast_2d(meanMisesLnV).T

u, s, vh = scipy.linalg.svd(d_MisesCauchy, full_matrices=False)

os.system('htop')
