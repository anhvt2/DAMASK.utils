import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic data for a 3D to 2D regression
# Suppose the relationship is y1 = 2*x1 - 3*x2 + 0.5*x3 and y2 = -x1 + 4*x2 - 0.2*x3
torch.manual_seed(42)
num_samples = 500
x = torch.randn(num_samples, 3)  # 3-dimensional input
y = torch.empty(num_samples, 2)  # 2-dimensional output
y[:, 0] = 2 * x[:, 0] - 3 * x[:, 1] + 0.5 * x[:, 2] + torch.randn(num_samples) * 0.1  # y1 with noise
y[:, 1] = -x[:, 0] + 4 * x[:, 1] - 0.2 * x[:, 2] + torch.randn(num_samples) * 0.1  # y2 with noise

# Define a simple regression model with 3D input and 2D output
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 5540),
        )

    def forward(self, x):
        return self.network(x)

# Instantiate the model, loss function, and optimizer
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 300
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(x)
    
    # Compute loss
    loss = criterion(y_pred, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plotting the results
# Sample some predictions for visualization
with torch.no_grad():
    y_pred = model(x)  # Make predictions on the input data

plt.figure(figsize=(12, 6))

# Plot predicted vs. true values for y1
plt.subplot(1, 2, 1)
plt.scatter(y[:, 0], y_pred[:, 0], alpha=0.6, label='Predicted vs Actual for y1')
plt.plot([y[:, 0].min(), y[:, 0].max()], [y[:, 0].min(), y[:, 0].max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel('True y1')
plt.ylabel('Predicted y1')
plt.legend()

# Plot predicted vs. true values for y2
plt.subplot(1, 2, 2)
plt.scatter(y[:, 1], y_pred[:, 1], alpha=0.6, label='Predicted vs Actual for y2')
plt.plot([y[:, 1].min(), y[:, 1].max()], [y[:, 1].min(), y[:, 1].max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel('True y2')
plt.ylabel('Predicted y2')
plt.legend()

plt.show()
