
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn as nn

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        file_name = self.data_frame.iloc[idx, 0]
        img_name = os.path.join(self.img_dir, file_name)
        image = Image.open(img_name).convert("RGB")  # Ensure images are in RGB mode

        if self.transform:
            image = self.transform(image)

        return image, file_name
    
# Transformations (optional)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
])

# Instantiate the dataset and DataLoader
root = '../../dataset'
csv_file = f'{root}/meta_data_for_encoder.csv'  # Replace with your CSV path
img_dir = f'{root}/removed_background'  # Replace with your image directory

# destination folder for encoded images
dst_dir = f'{root}/encoded_images'
os.makedirs(dst_dir, exist_ok=True)

# folder of saved model states
save_dir = "../../saved_models"
os.makedirs(save_dir, exist_ok=True)

dataset = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

print("Size of dataset: {}".format(len(dataset)))
print("Input image size: {}".format(dataset[0][0].size()))
print("Label name: {}".format(dataset[0][1]))

data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        
        # Encoder: Downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 2300x1800 -> 2300x1800
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2300x1800 -> 1150x900
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 1150x900 -> 1150x900
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1150x900 -> 575x450
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 575x450 -> 575x450
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 575x450 -> 287x225
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 287x225 -> 287x225
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((224, 224))  # 287x225 -> 224x224
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),  # 224x224 -> 224x224
            nn.ReLU()
        )
        
        # Decoder: Upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 128, kernel_size=3, stride=1, padding=1),  # 224x224 -> 224x224
            nn.ReLU(),
            nn.Upsample(size=(448, 448), mode='bilinear', align_corners=True),  # 224x224 -> 287x225
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # 287x225 -> 287x225
            nn.ReLU(),
            nn.Upsample(size=(1100, 900), mode='bilinear', align_corners=True),  # 287x225 -> 575x450
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # 575x450 -> 575x450
            nn.ReLU(),
            nn.Upsample(size=(2300, 1800), mode='bilinear', align_corners=True),  # 575x450 -> 2300x1800
            
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # 2300x1800 -> 2300x1800
            nn.Sigmoid()  # Ensure output is in range [0, 1]
        )

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Bottleneck
        bottleneck = self.bottleneck(encoded)
        # Decode
        decoded = self.decoder(bottleneck)
        return decoded

# Parameters
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_epochs = 100
learning_rate = 0.001

# Initialize the model
model = EncoderDecoder().to(device)
criterion = torch.nn.MSELoss()  # Assuming input is reconstructed as target
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_params)

# Training the model
model.train()
for epoch in range(num_epochs):
    for i, (inputs, _) in enumerate(data_loader):
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}")
    
    save_path = os.path.join(save_dir, f"encoder_model_{i}.pth")
    torch.save(model.state_dict(), save_path)

print("Training completed.")


# Save bottleneck outputs as images
model.load_state_dict(torch.load(save_path))
model.eval()
with torch.no_grad():
    for i, (inputs, file_name) in enumerate(data_loader):
        inputs = inputs.to(device)

        # Get bottleneck output
        bottleneck_features = model.bottleneck(model.encoder(inputs))
        
        # Normalize bottleneck features for visualization
        bottleneck_features = bottleneck_features - bottleneck_features.min()
        bottleneck_features = bottleneck_features / bottleneck_features.max()

        # Save each image in the batch
        for j in range(bottleneck_features.size(0)):
            feature = bottleneck_features[j].cpu()
            save_path = os.path.join(dst_dir, file_name[j])
            vutils.save_image(feature, save_path)
            
print(f"Bottleneck images saved in {dst_dir}.")
