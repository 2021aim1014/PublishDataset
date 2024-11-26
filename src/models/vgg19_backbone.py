import torch
import torch.nn as nn
from torchvision import models

class VGG19_backbone(nn.Module):
    def __init__(self, input_size=(3, 2300, 1800), output_size=4):
        super(VGG19_backbone, self).__init__()
        
        # Encoder: Downsampling
        self.encoder = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 2300x1800 -> 2300x1800
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # 2300x1800 -> 1150x900
            
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 1150x900 -> 1150x900
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # 1150x900 -> 575x450
            
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 575x450 -> 575x450
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # 575x450 -> 287x225
            
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),  # 287x225 -> 287x225
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((224, 224))  # 287x225 -> 224x224
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),  # 224x224 -> 224x224
            nn.ReLU()
        )

        # backbone: Load pre-trained VGG19 model
        self.model = models.vgg19(weights='IMAGENET1K_V1')

        # Freeze convolutional layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Modify the classifier
        self.model.classifier[6] = nn.Linear(4096, output_size)  # Output layer with the desired output size

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Bottleneck
        bottleneck = self.bottleneck(encoded)  
        # backbone and classifier
        output = self.model(bottleneck)
        return output

if __name__ == '__main__':
    batch_size = 4
    image_size = (3, 2300, 1800) 
    random_images = torch.randn(batch_size, *image_size)

    custom_model = VGG19_backbone(input_size=image_size)
    output = custom_model(random_images)
    print(f"Input Shape: {image_size}")
    print(f"Output Shape: {output.shape[1]}")
