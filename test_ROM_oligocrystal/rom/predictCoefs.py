import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

numFtrs  = 300 # number of ROM/POD features
fois     = ['MisesCauchy', 'MisesLnV'] # fields of interest
startIds = [0, 5540]

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

for foi, startId in zip(fois, startIds):

    class NNRegressor(nn.Module):
        def __init__(self):
            super(NNRegressor, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, numFtrs),
            )
        def forward(self, x):
            return self.network(x)

    # Function to load the model checkpoint
    def load_checkpoint(model, optimizer, foi):
        filename = "model_%s.pth" % foi
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return model, optimizer, start_epoch

    # Load trained model
    model = NNRegressor()
    model.double()
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, foi)

    y_train_pred = yscaler.inverse_transform(model(x_train).detach())
    y_test_pred  = yscaler.inverse_transform(model(x_test).detach())

    print(f'R^2 of POD coefs train for {foi} = {r2_score(y_train_pred.ravel(), y_train.ravel())}')
    print(f'R^2 of POD coefs test for {foi} = {r2_score(y_test_pred.ravel() , y_test.ravel())}')

    np.save('outputRom_Pred_%s' % foi, y_test_pred)

x_train = np.loadtxt('inputRom_Train.dat', delimiter=',', skiprows=1)
x_test  = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)
y_train = np.loadtxt('outputRom_Train.dat', delimiter=',', skiprows=1)
y_test  = np.loadtxt('outputRom_Test.dat',  delimiter=',', skiprows=1)

predCoefs_MisesCauchy = np.load('outputRom_Pred_MisesCauchy.npy')
predCoefs_MisesLnV    = np.load('outputRom_Pred_MisesLnV.npy')

predCoefs = np.hstack((predCoefs_MisesCauchy, predCoefs_MisesLnV))
headerStr = ['podCoef-MisesCauchy-%d' % i for i in range(1,numFtrs+1)] + ['podCoef-MisesLnV-%d' % i for i in range(1,numFtrs+1)] # output file header
header = ','.join(map(str, headerStr))

np.savetxt('outputRom_Pred.dat', predCoefs, delimiter=',', header=header, comments='')
