import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")  # Ensure images are in RGB mode
        labels = self.data_frame.iloc[idx, 1:].values.astype(float)  # Get all label columns
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels
    
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.CenterCrop(400),  # Start with a central crop (slightly larger than 224x224)
        transforms.RandomCrop(224),  # Randomly crop a 224x224 patch
        transforms.ToTensor(),  # Convert image to PyTorch tensor
    ])
    
    # --------------------- Train Dataset -------------------
    csv_file = '../dataset/meta_data_complete.csv'  # Replace with your CSV path
    img_dir = '../dataset/removed_background'  # Replace with your image directory
    dataset = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

    print("Size of dataset: {}".format(len(dataset)))
    print("Input size: {}".format(dataset[0][0].size()))
    print("Label size: {}".format(dataset[0][1].size()))

    # --------------------- Validate Dataset -------------------
    csv_file = '../dataset/validate_meta_data.csv'  # Replace with your CSV path
    img_dir = '../dataset/removed_background'  # Replace with your image directory
    dataset = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

    print("Size of dataset: {}".format(len(dataset)))
    print("Input size: {}".format(dataset[0][0].size()))
    print("Label size: {}".format(dataset[0][1].size()))
