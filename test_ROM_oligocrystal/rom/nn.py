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

numFtrs = 100 # number of ROM/POD features

x_train = np.loadtxt('inputRom_Train.dat', delimiter=',', skiprows=1)[:,:3]
y_train = np.loadtxt('outputRom_Train.dat', delimiter=',', skiprows=1)[:,:numFtrs]
x_test  = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)[:,:3]
y_test  = np.loadtxt('outputRom_Test.dat',  delimiter=',', skiprows=1)[:,:numFtrs]

# Take log of dotVarEps
x_train[:,0] = np.log10(x_train[:,0])
x_test[:,0]  = np.log10(x_test[:,0])

print(f'Elapsed time for loading datasets: {time.time() - t_start} seconds.')

# Reparameterize to convert a 3d -> 5540d problem to 4d -> 1d
def reparam(x,y):
    x = np.tile(x, [y.shape[1], 1])
    i = np.tile(np.atleast_2d(np.arange(y.shape[1])).T, [y.shape[0], 1])
    x = np.hstack((x,i))
    y = np.atleast_2d(y.ravel(order='C')).T
    return x,y

x_train, y_train = reparam(x_train, y_train)
x_test, y_test = reparam(x_test, y_test)

# Initialize the scaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# y_train /= 1.e9
# y_test /= 1.e9

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
            nn.Linear(4, 8),
            nn.ReLU6(),
            nn.Linear(8, 16),
            nn.ReLU6(),
            nn.Linear(16, 8),
            nn.ReLU6(),
            nn.Linear(8, 4),
            nn.ReLU6(),
            nn.Linear(4, 1),
        )
    def forward(self, x):
        return self.network(x)

# Function to load the model checkpoint
def load_checkpoint(model, optimizer, filename="model.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    return model, optimizer, start_epoch

# Instantiate the model, loss function, and optimizer
model = NNRegressor()
model.double()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
# scheduler = ExponentialLR(optimizer, gamma=1.05)  # Increase LR by 5% every epoch

# Lists to store training and test losses
train_losses = []
test_losses = []
test_id_losses = []
test_ood_loses = []

# Training loop
start_epoch = 0
num_epochs = 500000

try:
    model, optimizer, start_epoch = load_checkpoint(model, optimizer)
    print(f"Resuming training from epoch {start_epoch}...")
except FileNotFoundError:
    print("No saved model found. Starting training from scratch.")

for epoch in range(start_epoch, num_epochs):
    # Training phase
    model.train()
    y_train_pred = model(x_train)
    train_loss = criterion(y_train_pred, y_train)
    # Backward pass and optimization
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    # scheduler.step()
    # Evaluation phase (test set)
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        test_loss = criterion(y_test_pred, y_test)
    # Store losses for each epoch
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    # Print progress every 50 epochs
    # current_lr = optimizer.param_groups[0]['lr']
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")
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
# torch.save(model.state_dict(), 'model.pth')
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, 'model.pth')
print(f"Model saved to model.pth")

