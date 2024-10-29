import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Define transformations for the dataset (including normalization)
train_transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalizes to range [-1, 1] (mean=0.5, std=0.5)
])

# Load MNIST dataset
train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=train_transform)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # Convolutional Layer 1
        self.conv2 = nn.Conv2d(32, 64, 3)  # Convolutional Layer 2
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 10)  # Fully connected layer 2 (output layer)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.dropout = nn.Dropout(0.25)  # Dropout layer to avoid overfitting

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply Conv1, ReLU, and MaxPooling
        x = self.pool(torch.relu(self.conv2(x)))  # Apply Conv2, ReLU, and MaxPooling
        x = x.view(-1, 64 * 5 * 5)  # Flatten the output from conv layers
        x = torch.relu(self.fc1(x))  # Apply fully connected layer 1 with ReLU
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Final output layer
        return x


# Initialize model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Clear the gradients from the previous step
        output = model(X_batch)  # Forward pass
        loss = criterion(output, y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/10, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch)  # Get predictions
        _, predicted = torch.max(output, 1)  # Convert logits to class predictions
        total += y_batch.size(0)  # Count total samples
        correct += (predicted == y_batch).sum().item()  # Count correct predictions

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# Generate confusion matrix and classification report
y_pred, y_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        _, predicted = torch.max(output, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.cpu().numpy())

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))