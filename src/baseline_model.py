import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineModel(nn.Module):
    def __init__(self, input_size=(3, 224, 224), output_size=4, dropout_p = 0.8):
        super(BaselineModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        fc_input_dim = 128 * ((input_size[1] // 8) ** 2) 
        self.fc = nn.Linear(fc_input_dim, 64)  
        self.output = nn.Linear(64, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        # Convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for fully connected layers
        x = torch.flatten(x, start_dim=1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.output(x) 
        return x
    
# # Testing purpose
if __name__ == '__main__':
    batch_size = 1
    image_size = (3, 1024, 1024) 
    random_images = torch.randn(batch_size, *image_size)

    custom_model = BaselineModel(input_size=image_size)
    output = custom_model(random_images)
    print(f"Input Shape: {image_size}")
    print(f"Output Shape: {output.shape[1]}")

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    for params in custom_model.parameters():
        total_params += params.numel()
        if params.requires_grad:
            trainable_params += params.numel()

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non Trainable parameters: {non_trainable_params}")
