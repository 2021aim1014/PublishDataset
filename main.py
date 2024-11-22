import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
import pandas as pd

from src.image_dataset import ImageDataset
from src.plot import show_image_grid
from src.models.baseline_model import BaselineModel
from src.models.vgg19 import VGG19
from src.train import train_model

original_image_shape = (3, 224, 224)
image_shape = (3, 1800, 2300)

# Transformations (optional)
transform = transforms.Compose([
    transforms.Resize((image_shape[1], image_shape[2])),  # Resize to a common size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
])

# Instantiate the dataset and DataLoader
train_csv_file = 'dataset/meta_data_complete.csv'  # Replace with your CSV path
val_csv_file = 'dataset/validate_meta_data.csv'  # Replace with your CSV path
img_dir = 'dataset/removed_background'  # Replace with your image directory
train_dataset = ImageDataset(csv_file=train_csv_file, img_dir=img_dir, transform=transform)
val_dataset = ImageDataset(csv_file=val_csv_file, img_dir=img_dir, transform=transform)

print("Size of train dataset: {}".format(len(train_dataset)))
print("Size of test dataset: {}".format(len(val_dataset)))
print("Input test size: {}".format(val_dataset[0][0].size()))
print("Label test size: {}".format(val_dataset[0][1].size()))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

model_name = 'BaseLineModel_224'
model = BaselineModel(input_size=image_shape, output_size=4, dropout_p = 0.8)

# Loss function and optimizer
criterion = nn.MSELoss()  # For regression (use BCEWithLogitsLoss for multi-label classification)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming you have a Da taLoader ready as `data_loader`
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
num_epochs = 100
train_model(model_name, model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)
