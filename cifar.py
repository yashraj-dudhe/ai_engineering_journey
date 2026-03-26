import torch
import torch.nn as nn

class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,padding =1)
        self.bn1 =nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding =1)
        self.bn2 =nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()        
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding =1)
        self.bn3 =nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()        
        
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*8*8,512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512,10)
        
        
    def forward(self,x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        
        x = self.pool2(self.relu3(self.bn3(self.conv3(x))))
        
        
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
if __name__ == "__main__":
    dummy_image = torch.randn(1,3,32,32)
    model = CifarNet()
    output = model(dummy_image)
    print(f"model output shape: {output.shape}")
    print("VGG architecture ready")