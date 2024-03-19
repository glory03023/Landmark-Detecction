import torch
import torch.nn as nn
import torchvision.models as models

class LandmarkDetector(nn.Module):
    def __init__(self, num_landmarks):
        super(LandmarkDetector, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048, num_landmarks * 2)  # Assuming each landmark has x and y coordinates

    def forward(self, x):
        features = self.resnet(x)
        landmarks = self.fc(features)
        return landmarks

# Example usage:
if __name__ == "__main__":
    # Instantiate the landmark detector model
    num_landmarks = 10  # Example: detecting 10 landmarks
    model = LandmarkDetector(num_landmarks)

    # Example input tensor (batch_size, channels, height, width)
    batch_size = 1
    channels = 3
    height = 224
    width = 224
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Forward pass
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Should be (batch_size, num_landmarks * 2)