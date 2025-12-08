import torch
import torch.nn as nn


class MNSITClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, padding = 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride =2)
        
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 32, kernel_size = 3, padding = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride =2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features = 32*7*7, out_features = 10)
        
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))        
        
        x = self.flatten(x)
        
        logits = self.fc1(x)
        return logits
    

if __name__ == "__main__":
    model = MNSITClassifier()
    
    dummy_image = torch.randn(1,1,28,28)
    
    output = model(dummy_image)
    
    print(f"model structure:\n {model}")
    print(f"input shape:\n {dummy_image.shape}")
    print(f"output shape:\n {output.shape}")
    print(f"Raw output:\n {output}")