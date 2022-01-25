
import numpy as np
import os, sys, time, glob, datetime

# read input from dummy dakota output (with the same dimensionality)
sg_input_list = np.loadtxt('dakota_sparse_tabular.dat', skiprows=1)


matcfgFile = open('material.config') # use 'materials.config' as the template file
txtcfg = matcfgFile.readlines()
matcfgFile.close()


