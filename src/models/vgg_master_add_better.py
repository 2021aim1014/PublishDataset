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
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

          # Adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Adjust input size based on your feature map
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        # Forward pass through VGG19 blocks
        vgg_block1_out = self.backbone_features[:5](x)  # Corresponds to block1_pool
        vgg_block2_out = self.backbone_features[5:10](vgg_block1_out)  # block2_pool
        vgg_block3_out = self.backbone_features[10:19](vgg_block2_out)  # block3_pool
        vgg_block4_out = self.backbone_features[19:28](vgg_block3_out)  # block4_pool
        vgg_block5_out = self.backbone_features[28:](vgg_block4_out)  # block5_pool

        # Forward pass through custom network
        x = self.conv_block1(x)
        x = (x + vgg_block1_out)/2

        x = self.conv_block2(x)
        x = (x + vgg_block2_out)/2

        x = self.conv_block3(x)
        x = (x + vgg_block3_out)/2

        x = self.conv_block4(x)
        x = (x + vgg_block4_out)/2

        x = self.conv_block5(x)
        x = (x + vgg_block5_out)/2
        
        # Adaptive average pooling
        x = self.avgpool(x)

        # Flatten before classifier
        x = torch.flatten(x, 1)

        # Classifier
        x = self.classifier(x)

        return x

