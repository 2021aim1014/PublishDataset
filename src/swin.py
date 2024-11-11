import torch
from transformers import SwinModel, SwinConfig, AutoImageProcessor
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from data_loader import ImageMultiLabelDataset
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def model_details(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - total_trainable_params

    print(f'Details about swin {name}')
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {total_trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

model_name = "microsoft/swin-base-patch4-window7-224"
backbone = SwinModel.from_pretrained(model_name).to(device)
for param in backbone.parameters():
    param.requires_grad = False     # Freeze all parameters in the backbone model
  
feature_extractor = AutoImageProcessor.from_pretrained(model_name)
model_details(backbone, 'Backbone')

class SwinForRegression(nn.Module):
    def __init__(self, backbone, output_dim=4):
        super(SwinForRegression, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(backbone.config.hidden_size, 1024)
        self.regression_head = nn.Linear(1024, output_dim)
   
    def forward(self, x):
        features = self.backbone(x).pooler_output  # Extract features from the last layer
        output = self.fc1(features)
        output = self.regression_head(output)    # Apply regression head
        return output


# Instantiate the regression model
model = SwinForRegression(backbone, output_dim=4).to(device)
model_details(model, 'Swin finetune')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

augmented_transform = transforms.Compose([
    transforms.Resize((224, 224)),             # Resize to expected input size
    transforms.RandomRotation(degrees=30),     # Randomly rotate images within 30 degrees
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)), # Random zoom in/out by cropping
    transforms.RandomHorizontalFlip(),         # Flip horizontally with 50% probability
    transforms.RandomVerticalFlip(),           # Flip vertically with 50% probability
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Adjust colors
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Instantiate the dataset and DataLoader
csv_file = '../dataset/meta_data_complete.csv'  # Replace with your CSV path
img_dir = '../dataset/removed_background'  # Replace with your image directory
dataset = ImageMultiLabelDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

# Assuming the dataset is already created
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation

# Split dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
criterion = nn.MSELoss()

def train_model(model, train_dataset, criterion, num_epochs = 10):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device).float()
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient Clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()  # Adjust learning rate
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader):.4f}")

    return model

def validate_model(model, val_loader, criterion):
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device).float()
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        
    print(f"Epoch {101}/{101}, Val Loss: {val_loss / len(val_loader):.4f}")

num_epochs = 100
model = train_model(model, train_dataset, criterion)
validate_model(model, val_dataset, criterion)

# Check model after image transformation to check whether model is able to generalize or not
dataset = ImageMultiLabelDataset(csv_file=csv_file, img_dir=img_dir, transform=augmented_transform)
# Split dataset into training and validation sets
_, val_dataset = random_split(dataset, [train_size, val_size])
validate_model(model, val_dataset, criterion)
