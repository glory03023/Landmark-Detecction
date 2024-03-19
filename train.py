import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import LandmarkDetector
from dataset import LandmarkDataset

# Define hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_landmarks = 10  # Example: 10 landmarks

# Create dataset and dataloaders
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = LandmarkDataset(data_dir='dataset/train', transform=transform, shuffle=True)  # Define your dataset class
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the model
model = LandmarkDetector(num_landmarks)  # Assuming you have a LandmarkDetector class defined

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data  # Assuming your dataset returns inputs and labels

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

print('Finished Training')