import torch
import torch.nn as nn
import torchvision.models as models

class VGG19Master(nn.Module):
    def __init__(self, num_classes=4, img_shape=(3, 224, 224)):
        super(VGG19Master, self).__init__()

        # Load pre-trained VGG19 model
        backbone = models.vgg19(weights='IMAGENET1K_V1')
        self.backbone_features = backbone.features


        # Freeze convolutional layers
        for param in self.backbone_features.parameters():
            param.requires_grad = False

        # Define custom convolution blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024 * 7 * 7, 1024)  # Adjust input size for your task
        self.output_layer = nn.Linear(1024, num_classes)  # Adjust output size for your task

    def forward(self, x):
        # Forward pass through VGG19 blocks
        vgg_block1_out = self.backbone_features[:5](x)  # Corresponds to block1_pool
        vgg_block2_out = self.backbone_features[5:10](vgg_block1_out)  # block2_pool
        vgg_block3_out = self.backbone_features[10:19](vgg_block2_out)  # block3_pool
        vgg_block4_out = self.backbone_features[19:28](vgg_block3_out)  # block4_pool
        vgg_block5_out = self.backbone_features[28:](vgg_block4_out)  # block5_pool

        # Forward pass through custom network
        x = self.conv_block1(x)
        x = torch.cat((x, vgg_block1_out), dim=1)

        x = self.conv_block2(x)
        x = torch.cat((x, vgg_block2_out), dim=1)

        x = self.conv_block3(x)
        x = torch.cat((x, vgg_block3_out), dim=1)

        x = self.conv_block4(x)
        x = torch.cat((x, vgg_block4_out), dim=1)

        x = self.conv_block5(x)
        x = torch.cat((x, vgg_block5_out), dim=1)

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        return self.output_layer(x)

if __name__ == '__main__':
    
    batch_size = 1
    image_size = (3, 224, 224) 
    random_images = torch.randn(batch_size, *image_size)

    custom_model = VGG19Master(img_shape=image_size)
    output = custom_model(random_images)
    print(f"Input Shape: {image_size}")
    print(f"Output Shape: {output.shape[1]}")

    # Initialize parameter counters
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    # Count parameters
    for params in custom_model.parameters():
        total_params += params.numel()
        if params.requires_grad:
            trainable_params += params.numel()
        else:
            non_trainable_params += params.numel()

    # Print parameter counts
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
