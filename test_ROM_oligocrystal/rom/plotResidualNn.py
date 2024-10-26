import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

# Sample log data (you would typically read this from a file)
with open('training_log.txt', 'r') as file:
    log_data = file.read()

# Parse the log data
epochs = []
train_losses = []
test_losses = []

for line in log_data.strip().split('\n'):
    parts = line.split(',')
    epoch = int(parts[0].split('[')[1].split('/')[0])  # Extract epoch number
    train_loss = float(parts[1].split(':')[1].strip())  # Extract train loss
    test_loss = float(parts[2].split(':')[1].strip())   # Extract test loss
    
    epochs.append(epoch)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# Convert to numpy arrays for easier manipulation
epochs = np.array(epochs)
train_losses = np.array(train_losses)
test_losses = np.array(test_losses)

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label=r'Train Loss', marker='o', c="tab:blue")
plt.plot(epochs, test_losses, label=r'Test Loss', marker='s', c="tab:orange")
plt.xlabel('Epochs', fontsize=24)
plt.ylabel('Loss', fontsize=24)
plt.title('Train/Test Loss over Epochs', fontsize=24)
plt.yscale('log')  # Using logarithmic scale for better visibility
plt.legend(fontsize=24, loc='best', markerscale=2.5)
# plt.grid()
plt.show()
