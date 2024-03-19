import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import LandmarkDataset  # Assuming you have a dataset class defined

# Define evaluation metrics
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

# Define hyperparameters
batch_size = 32
num_landmarks = 10  # Example: 10 landmarks

# Create dataset and dataloaders
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
test_dataset = LandmarkDataset(data_dir='dataset/test', transform=transform)  # Define your dataset class
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model = torch.load('path/to/trained/model.pth')  # Replace 'path/to/trained/model.pth' with the path to your trained model file

# Define loss function
criterion = nn.MSELoss()

# Evaluate the model
test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}')