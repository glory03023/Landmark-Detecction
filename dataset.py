import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class LandmarkDataset(Dataset):
    def __init__(self, data_dir, transform=None, shuffle=False):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.landmark_coords = []
        
        # Load data from directory
        self._load_data()

        # Optionally shuffle data
        if self.shuffle:
            self._shuffle_data()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        landmark_coords = self.landmark_coords[idx]

        # Read image
        image = Image.open(image_path)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Convert landmark coordinates to tensor
        landmark_coords = torch.tensor(landmark_coords, dtype=torch.float32)

        return image, landmark_coords
    
    def _load_data(self):
        # Iterate through data directory
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Extract image path
                image_path = os.path.join(self.data_dir, filename)
                self.image_paths.append(image_path)

                # Extract landmark coordinates from filename or label file
                landmark_coords = self._extract_landmark_coords(filename)
                self.landmark_coords.append(landmark_coords)

    def _extract_landmark_coords(self, filename):
        # Example: Parse filename to extract landmark coordinates
        # For demonstration, assume filename format is: image_x1_y1_x2_y2_..._xn_yn.jpg
        coords = filename.split("_")[1:-1]  # Exclude "image" and ".jpg"
        landmark_coords = [int(coord) for coord in coords]
        return landmark_coords

    def _shuffle_data(self):
        # Shuffle image paths and landmark coordinates in unison
        combined = list(zip(self.image_paths, self.landmark_coords))
        torch.manual_seed(0)  # Set random seed for reproducibility
        torch.random.shuffle(combined)
        self.image_paths, self.landmark_coords = zip(*combined)

