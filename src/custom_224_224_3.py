import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_loader import ImageMultiLabelDataset
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def model_details(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - total_trainable_params

    print(f'Details about {name}')
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {total_trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4, img_shape=(3, 224, 224)):
        super(SimpleCNN, self).__init__()
       
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
       
        # Pooling layer: since no training required, hence sharing is ok
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
       
        # Fully connected layers
        self.flat_total = 64 * ((img_shape[1] // 8) ** 2)  # channels * (224 // #of conv)**2
        self.fc1 = nn.Linear(self.flat_total, 64)  # Assuming input images are resized to (128, 128)
        self.fc2 = nn.Linear(64, num_classes)   # Output layer for 4 labels
       
        # Activation function: since no training required, hence sharing is ok
        self.relu = nn.ReLU()
   
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (3, 224, 224) -> Conv1(@16) -> ReLU -> MaxPool -> (16, 112, 112)
        x = self.pool(self.relu(self.conv2(x)))  # (16, 112, 112) -> Conv2(@32) -> ReLU -> MaxPool -> (32, 56, 56)
        x = self.pool(self.relu(self.conv3(x)))  # (32, 56, 56) -> Conv3(@64) -> ReLU -> MaxPool -> (64, 28, 28)
       
        x = x.view(-1, self.flat_total)  # Flatten the tensor
        x = self.relu(self.fc1(x))    # Fully connected layer 1 -> ReLU
        x = self.fc2(x)               # Output layer (for 4 labels)
        return x
    
# Instantiate the regression model
model = SimpleCNN(num_classes=4, img_shape=(3, 224, 224)).to(device)
model_details(model, 'Custom')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

augmented_transform = transforms.Compose([
    transforms.Resize((224, 224)),             # Resize to expected input size
    transforms.RandomRotation(degrees=30),     # Randomly rotate images within 30 degrees
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)), # Random zoom in/out by cropping
    transforms.RandomHorizontalFlip(),         # Flip horizontally with 50% probability
    transforms.RandomVerticalFlip(),           # Flip vertically with 50% probability
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Adjust colors
    transforms.ToTensor(),
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

def train_model(model, train_dataset, criterion):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    num_epochs = 1

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

model = train_model(model, train_dataset, criterion)

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

validate_model(model, val_dataset, criterion)

dataset = ImageMultiLabelDataset(csv_file=csv_file, img_dir=img_dir, transform=augmented_transform)
_, val_dataset = random_split(dataset, [train_size, val_size])
validate_model(model, val_dataset, criterion)


