import numpy as np
import os, sys
import time
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 24
SolidIdx = np.loadtxt('SolidIdx.dat', dtype=int)

fois = ['MisesCauchy', 'MisesLnV'] # fields of interest
labels = [r'$\sigma_{vM}$', r'$\varepsilon_{vM}$']
# fois = ['MisesCauchy'] # fields of interest
# fois = ['MisesLnV'] # fields of interest

t_start = time.time()

foi = 'MisesCauchy'
# Load data
d = np.load(f'd_{foi}.npy') # Original data

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Linear(input_dim, latent_dim)
        # Decoder
        self.decoder = nn.Linear(latent_dim, input_dim)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Function to train the autoencoder
def train_autoencoder(data, latent_dim=10, epochs=100, lr=0.001, lambda_reg=4e0):
    input_dim = data.shape[0]  # Each column is a sample
    data = torch.tensor(data, dtype=torch.float32).T  # Transpose to have samples as rows
    
    model = Autoencoder(input_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        reconstruction_loss = criterion(output, data)
        frobenius_norm = torch.norm(model.encoder.weight, p='fro') * torch.norm(model.decoder.weight, p='fro')
        loss = reconstruction_loss + lambda_reg * frobenius_norm
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return model

# Function to encode new data
def encode_data(model, data):
    data = torch.tensor(data, dtype=torch.float32).T
    with torch.no_grad():
        encoded = model.encoder(data)
    return encoded.numpy().T  # Transpose back to match input format
