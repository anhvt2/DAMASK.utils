import numpy as np
import pandas as pd

d = 5 # d: dimensionality

# np.savetxt('input.dat', list(np.random.rand(6,1)), fmt='%.18e')
# pd.DataFrame(np.random.rand(6,1)).to_csv('input.dat')
np.random.rand(d,1).tofile('input.dat', sep=',', format='%.18e')
