import gpflow
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter
import argparse
import numpy.matlib


import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'nn.py.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

t_start = time.time()

# Get device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

numFtrs = 10 # number of ROM/POD features

x_train = np.loadtxt('inputRom_Train.dat', delimiter=',', skiprows=1)[:,[0,1,4]]
y_train = np.loadtxt('outputRom_Train.dat', delimiter=',', skiprows=1)[:,:numFtrs]
x_test  = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)[:,[0,1,4]]
y_test  = np.loadtxt('outputRom_Test.dat',  delimiter=',', skiprows=1)[:,:numFtrs]

# Take log of dotVarEps
x_train[:,0] = np.log10(x_train[:,0])
x_test[:,0]  = np.log10(x_test[:,0])
x_train[:,2] = np.log2(x_train[:,2])
x_test[:,2]  = np.log2(x_test[:,2])

print(f'Elapsed time for loading datasets: {time.time() - t_start} seconds.')

