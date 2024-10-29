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

# Get device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

numFtrs = 200 # number of ROM/POD features

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

# Standardize datasets
xscaler = StandardScaler()
xscaler.fit(x_train)
x_train_scaled = xscaler.transform(x_train)
x_test_scaled  = xscaler.transform(x_test)

yscaler = StandardScaler()
yscaler.fit(y_train)
y_train_scaled = yscaler.transform(y_train)
y_test_scaled  = yscaler.transform(y_test)
weights = torch.from_numpy(np.sqrt(yscaler.var_ / yscaler.var_.min()))

# # Reparameterize to convert a 3d -> 5540d problem to 4d -> 1d
# def reparam(x,y):
#     x = np.tile(x, [y.shape[1], 1])
#     i = np.tile(np.atleast_2d(np.arange(y.shape[1])).T, [y.shape[0], 1])
#     x = np.hstack((x,i))
#     y = np.atleast_2d(y.ravel(order='C')).T
#     return x,y

# x_train, y_train_scaled  = reparam(x_train, y_train_scaled)
# x_test , y_test_scaled   = reparam(x_test , y_test_scaled)

# Convert to torch format
x_train = torch.from_numpy(x_train)
x_test  = torch.from_numpy(x_test)
y_train_scaled = torch.from_numpy(y_train_scaled)
y_test_scaled  = torch.from_numpy(y_test_scaled)

# Define a multi-layer neural network with non-linear activation functions
class NNRegressor(nn.Module):
    def __init__(self):
        super(NNRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, numFtrs),
        )
    def forward(self, x):
        return self.network(x)

# Random weight initialization: Xavier Initialization for Linear layers
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)  # Initialize biases with zeros

# Function to load the model checkpoint
def load_checkpoint(model, optimizer, filename="model3d.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    return model, optimizer, start_epoch

# Custom Weighted MSE Loss function
def WeightedMSELoss(predicted, target, weights):
    squared_difference = (predicted - target) ** 2
    weighted_squared_difference = weights * squared_difference
    weighted_mse = weighted_squared_difference.mean()
    return weighted_mse

# Instantiate the model, loss function, and optimizer
model = NNRegressor()
model.double()
initialize_weights(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store training and test losses
train_losses = []
test_losses = []

# Training loop
start_epoch = 0
num_epochs = 1000000

try:
    model, optimizer, start_epoch = load_checkpoint(model, optimizer)
    print(f"Resuming training from epoch {start_epoch}...")
except FileNotFoundError:
    print("No saved model found. Starting training from scratch.")

y_train_pred = yscaler.inverse_transform(model(x_train).detach())
y_test_pred  = yscaler.inverse_transform(model(x_test).detach())

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

print(f'R^2 (POD coefs) test dataset = {r2_score(y_train_pred.ravel(), y_train.ravel())}')
print(f'R^2 (POD coefs) test dataset = {r2_score(y_test_pred.ravel() , y_test.ravel())}')
