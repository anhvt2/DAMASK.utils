
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Get device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Generate sample data with a non-linear relationship
torch.manual_seed(42)
x = torch.linspace(-10, 10, 200).reshape(-1, 1)
y = 0.5 * x**3 - 3 * x**2 + 2 * x + torch.randn(x.size()) * 10  # Non-linear data with noise

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define a multi-layer neural network with non-linear activation functions
class NonlinearRegressionModel(nn.Module):
    def __init__(self):
        super(NonlinearRegressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# Instantiate the model, loss function, and optimizer
model = NonlinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Lists to store training and test losses
train_losses = []
test_losses = []

# Training loop
num_epochs = 500
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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Plotting the train and test loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Train/Test Loss Convergence')
plt.legend()

# Plotting the results
plt.figure(figsize=(10, 5))
predicted = model(x).detach()  # Detach for plotting
plt.scatter(x, y, label='Original Data')
plt.plot(x, predicted, color='red', label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
