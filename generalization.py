import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import copy
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.image_dataset import ImageDataset
from src.plot import show_image_grid
from src.models.baseline_model import BaselineModel
from src.models.vgg19 import VGG19
from src.models.vgg19_master_add import VGG19Master as VGG19Master_ADD
from src.models.vgg19_master_concat import VGG19Master as VGG19Master_CONCAT
from src.models.vgg19_backbone import VGG19_backbone
from src.models.vgg19_master_add_better import VGG19Master as VGG19MasterBetter

from torchvision import transforms
from torchvision.transforms import RandomAffine, RandomHorizontalFlip

image_shape = (3, 224, 224)
val_csv_file = 'dataset/validate_meta_data.csv'  # Replace with your CSV path
img_dir = 'dataset/removed_background'  # Replace with your image directory

# Define a list of transformations
transformations = [
    transforms.Compose([  
        transforms.Resize((image_shape[1], image_shape[2])),  # Resize to a common size
        transforms.RandomRotation(degrees=45)             # Rotation
    ]),

    transforms.Compose([  
        transforms.Resize((image_shape[1], image_shape[2])),  # Resize to a common size
        transforms.RandomResizedCrop(size=(224, 224))       # Random Crop
    ]),

    transforms.Compose([  
        transforms.Resize((image_shape[1], image_shape[2])),  # Resize to a common size
        transforms.RandomHorizontalFlip(p=1.0)             # Horizontal Flip
    ]),

    transforms.Compose([  
        transforms.Resize((image_shape[1], image_shape[2])),  # Resize to a common size
        transforms.RandomVerticalFlip(p=1.0)               # Vertical Flip
    ]),

    transforms.Compose([  
        transforms.Resize((image_shape[1], image_shape[2])),  # Resize to a common size
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)  # Color Jitter
    ]),

    transforms.Compose([  
        transforms.Resize((image_shape[1], image_shape[2])),  # Resize to a common size
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))  # Gaussian Blur
    ]),

    transforms.Compose([  
        transforms.Resize((image_shape[1], image_shape[2])),  # Resize to a common size
        transforms.Grayscale(num_output_channels=3)        # Grayscale
    ]),

    transforms.Compose([      
        transforms.Resize((image_shape[1], image_shape[2])),  # Resize to a common size                          # Composite Transformations
        RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        RandomHorizontalFlip()
    ]),

    transforms.Compose([
        transforms.Resize((image_shape[1], image_shape[2])),  # Resize to a common size
        RandomAffine(degrees=30, translate=(0.2, 0.2)),
        transforms.ColorJitter(brightness=0.5),
        RandomHorizontalFlip(),
        transforms.GaussianBlur(5)
    ]),

    transforms.Compose([
        transforms.Resize((image_shape[1], image_shape[2])),  # Resize to a common size
        transforms.RandomRotation(degrees=45),             # Rotation
        transforms.RandomHorizontalFlip(p=1.0),             # Horizontal Flip
        RandomHorizontalFlip(),
        transforms.GaussianBlur(5)
    ])

]

# Instantiate the dataset and DataLoader
val_dataset = ImageDataset(csv_file=val_csv_file, img_dir=img_dir, transform=transforms.ToTensor())

# Function to apply transformations
def apply_transformation(dataset, transform):
    # Copy the dataset and apply a new transformation
    dataset = copy.deepcopy(dataset)
    dataset.transform = transforms.Compose([transform, transforms.ToTensor()])
    return dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'BaseLineModel_224'
model = BaselineModel(input_size=image_shape, output_size=4, dropout_p = 0.8)
model.load_state_dict(torch.load(f"saved_models/{model_name}.pth"))
model.to(device)
model.eval()

# Evaluate the model on transformed datasets
results = {}

with torch.no_grad():  # No gradient computation
    for i, transform in enumerate(transformations):
        print(f"Evaluating transformation {i+1}/{len(transformations)}")

        # Apply transformation to dataset
        transformed_dataset = apply_transformation(val_dataset, transform)
        dataloader = DataLoader(transformed_dataset, batch_size=8, shuffle=False, num_workers=4)

        y_true = []
        y_pred = []

        # Pass images through the model
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Collect predictions and labels
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            break

        # Calculate metrics
        y_true = torch.tensor(np.array(y_true)).squeeze()
        y_pred = torch.tensor(np.array(y_pred)).squeeze()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Store metrics
        results[f"Transformation {i+1}"] = {'MSE': mse, 'MAE': mae, 'R2': r2}
        print(f"Transformation {i+1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Convert the nested dictionary to a DataFrame
results_df = pd.DataFrame(results).T  # Transpose to make transformations as rows

# Save the DataFrame to a CSV file
results_df.to_csv(f'results/{model_name}_val.csv', index_label='Transformation')
