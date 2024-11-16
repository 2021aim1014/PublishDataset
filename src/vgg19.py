import torch
import torch.nn as nn
from torchvision import models

class VGG19(nn.Module):
    def __init__(self, input_size=(3, 224, 224), output_size=4, finetune=False):
        super(VGG19, self).__init__()
        
        # Load pre-trained VGG19 model
        model = models.vgg19(weights='IMAGENET1K_V1')

        # Freeze convolutional layers
        for param in model.features.parameters():
            param.requires_grad = finetune

        # Modify the classifier
        model.classifier[6] = nn.Linear(4096, output_size)  # Output layer with the desired output size
        self.model = model

    def forward(self, x):
        return self.model(x)


# # Testing purpose
if __name__ == '__main__':
    batch_size = 1
    image_size = (3, 224, 224) 
    random_images = torch.randn(batch_size, *image_size)

    custom_model = VGG19(input_size=image_size, finetune=False)
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
