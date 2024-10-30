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
logFileName = 'predictCoefs.py.log'
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

class NNRegressor_MisesCauchy(nn.Module):
    def __init__(self):
        super(NNRegressor_MisesCauchy, self).__init__()
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

class NNRegressor_MisesLnV(nn.Module):
    def __init__(self):
        super(NNRegressor_MisesLnV, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 16),
            nn.Sigmoid(),
            nn.Linear(16, 32),
            nn.Sigmoid(),
            nn.Linear(32, 64),
            nn.Sigmoid(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, numFtrs),
        )
    def forward(self, x):
        return self.network(x)

# Function to load the model checkpoint
# def load_checkpoint(model, optimizer, foi):
def load_checkpoint(model, foi):
    filename = "model_%s.pth" % foi
    checkpoint = torch.load(filename)
    print(f'Loading model in {filename}')
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch'] + 1
    # return model, optimizer, start_epoch
    return model

for foi, startId, model in zip(fois, startIds, [NNRegressor_MisesCauchy(), NNRegressor_MisesLnV()]):
    x_train = np.loadtxt('inputRom_Train.dat', delimiter=',', skiprows=1)[:,[0,1,4]]
    x_test  = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)[:,[0,1,4]]
    y_train = np.loadtxt('outputRom_Train.dat', delimiter=',', skiprows=1)[:,startId:startId+numFtrs]
    y_test  = np.loadtxt('outputRom_Test.dat',  delimiter=',', skiprows=1)[:,startId:startId+numFtrs]
    # Take log of dotVarEps
    x_train[:,0] = np.log10(x_train[:,0])
    x_test[:,0]  = np.log10(x_test[:,0])
    x_train[:,2] = np.log2(x_train[:,2])
    x_test[:,2]  = np.log2(x_test[:,2])
    # Standardize datasets
    xscaler = StandardScaler()
    xscaler.fit(x_train)
    x_train_scaled = xscaler.transform(x_train)
    x_test_scaled  = xscaler.transform(x_test)
    # Scale output
    yscaler = StandardScaler()
    yscaler.fit(y_train)
    y_train_scaled = yscaler.transform(y_train)
    y_test_scaled  = yscaler.transform(y_test)
    # Convert numpy to torch
    x_train = torch.from_numpy(x_train)
    x_test  = torch.from_numpy(x_test)
    y_train_scaled = torch.from_numpy(y_train_scaled)
    y_test_scaled  = torch.from_numpy(y_test_scaled)
    # Load trained model
    model.double()
    model = load_checkpoint(model, foi)
    y_train_pred = yscaler.inverse_transform(model(x_train).detach())
    y_test_pred  = yscaler.inverse_transform(model(x_test).detach())
    print(f'R^2 of POD coefs train for {foi} = {r2_score(y_train_pred.ravel(), y_train.ravel())}')
    print(f'R^2 of POD coefs test for {foi} = {r2_score(y_test_pred.ravel() , y_test.ravel())}')
    np.save('outputRom_Pred_%s' % foi, y_test_pred)
    print(f'Elapsed time: {time.time() - t_start} seconds.')


predCoefs_MisesCauchy = np.load('outputRom_Pred_MisesCauchy.npy')
predCoefs_MisesLnV    = np.load('outputRom_Pred_MisesLnV.npy')

predCoefs = np.hstack((predCoefs_MisesCauchy, predCoefs_MisesLnV))
headerStr = ['podCoef-MisesCauchy-%d' % i for i in range(1,numFtrs+1)] + ['podCoef-MisesLnV-%d' % i for i in range(1,numFtrs+1)] # output file header
header = ','.join(map(str, headerStr))

np.savetxt('outputRom_Pred.dat', predCoefs, delimiter=',', header=header, comments='')
print(f'Elapsed time for dumping predicted POD coefs: {time.time() - t_start} seconds.')

print(f'Dumping predicted POD coefs to local folders...')
time.sleep(3)

# Write local predicted POD coefs in 
x_test       = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)
DamaskIdxs   = x_test[:,5].astype(int)
PostProcIdxs = x_test[:,6].astype(int)

numFolders = len(DamaskIdxs)

for i in range(numFolders):
    outFileName = '../damask/%d/postProc/predPodCoefs_main_tension_inc%s' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2))
    tmpCoefs = np.zeros([5540,2])
    tmpCoefs[:numFtrs,0] = predCoefs_MisesCauchy[i,:]
    tmpCoefs[:numFtrs,1] = predCoefs_MisesLnV[i,:]
    np.save(outFileName, tmpCoefs)
    logging.info(f'Processing {i+1:<d}/{numFolders} folders: dumped {outFileName}')

logging.info(f'Finish dumping local POD coefs in {time.time() - t_start} seconds.')

