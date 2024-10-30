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

numFtrs  = 300 # number of ROM/POD features
fois     = ['MisesLnV'] # fields of interest
startIds = [5540]

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


for foi, startId in zip(fois, startIds):

    x_train = np.loadtxt('inputRom_Train.dat', delimiter=',', skiprows=1)[:,[0,1,4]]
    x_test  = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)[:,[0,1,4]]
    y_train = np.loadtxt('outputRom_Train.dat', delimiter=',', skiprows=1)[:,startId:startId+numFtrs]
    y_test  = np.loadtxt('outputRom_Test.dat',  delimiter=',', skiprows=1)[:,startId:startId+numFtrs]

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
    # weights = torch.from_numpy(np.sqrt(yscaler.var_ / yscaler.var_.min()))
    eigenvalues = np.load('podEigen_%s.npy' % foi)
    weights = torch.from_numpy(eigenvalues[:numFtrs] / np.min(eigenvalues[:numFtrs]))

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

    # Random weight initialization: Xavier Initialization for Linear layers
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)  # Initialize biases with zeros

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
    num_epochs = 100000

    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        y_train_pred_scaled = model(x_train)
        train_loss = WeightedMSELoss(y_train_pred_scaled, y_train_scaled, weights)
        # Backward pass and optimization
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # Evaluation phase (test set)
        model.eval()
        with torch.no_grad():
            y_test_pred_scaled = model(x_test)
            test_loss = WeightedMSELoss(y_test_pred_scaled, y_test_scaled, weights)
        # Store losses for each epoch
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        if (epoch + 1) % 50 == 0:
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
        if (epoch + 1) % 5000 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'model_%s.pth' % foi)
            print(f"Model saved to model_{foi}.pth")

    y_train_pred = yscaler.inverse_transform(model(x_train).detach())
    y_test_pred  = yscaler.inverse_transform(model(x_test).detach())

    print(f'R^2 of POD coefs train for {foi} = {r2_score(y_train_pred.ravel(), y_train.ravel())}')
    print(f'R^2 of POD coefs test for {foi} = {r2_score(y_test_pred.ravel() , y_test.ravel())}')

    np.save('outputRom_Pred_%s' % foi, y_test_pred)

