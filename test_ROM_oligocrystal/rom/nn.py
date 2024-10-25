import os
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

x_train = np.loadtxt('inputRom_Train.dat', delimiter=',', skiprows=1)[:,:3]
x_test  = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)[:,:3]
y_train = np.loadtxt('outputRom_Train.dat', delimiter=',', skiprows=1)[:,:5540]
y_test  = np.loadtxt('outputRom_Test.dat',  delimiter=',', skiprows=1)[:,:5540]

logging.info(f'Elapsed time for loading datasets: {time.time() - t_start} seconds.')

# Initialize the scaler
scaler = StandardScaler()
# Fit the scaler on the training data
scaler.fit(x_train)
# Transform both training and test data
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Convert to torch format
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# Define a multi-layer neural network with non-linear activation functions
class NNRegressor(nn.Module):
    def __init__(self):
        super(NNRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 500),
            nn.Tanh(),
            nn.Linear(500, 1000),
            nn.Tanh(),
            nn.Linear(1000, 5540),
        )
    def forward(self, x):
        return self.network(x)

# Instantiate the model, loss function, and optimizer
model = NNRegressor()
model.double()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Lists to store training and test losses
train_losses = []
test_losses = []

# Training loop
num_epochs = 50000
for epoch in range(num_epochs):
    # Training phase
    model.train()
    y_train_pred = model(x_train)
    train_loss = criterion(y_train_pred, y_train)
    # Backward pass and optimization
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    # Evaluation phase (test set)
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        test_loss = criterion(y_test_pred, y_test)
    # Store losses for each epoch
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# # Plotting the train and test loss curves
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Train Loss')
# plt.plot(test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.title('Train/Test Loss Convergence')
# plt.legend()

# Save the model state dictionary
torch.save(model.state_dict(), 'model.pth')
print(f"Model saved to model.pth")

