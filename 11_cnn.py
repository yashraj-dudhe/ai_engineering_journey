import torch
import torch.nn as nn

def cnn_drill():
    print("--1. The setup----")
    
    input_image = torch.randn(1,1,28,28)
    print(f"Input shape: {input_image.shape}")
    
    print("---2. The convolutional layer ---")
    conv = nn.Conv2d(in_channels  =1, out_channels =8,kernel_size =3, padding = 1)
    
    x = conv(input_image)
    print(f"Image shape after convulutional layer {x.shape}")
    
    print("the activation relu")
    relu = nn.ReLU()
    x = relu(x)
    
    print("---3. Max pooling ----")
    pool = nn.MaxPool2d(kernel_size=2,stride = 2)
    x = pool(x)
    print(f"after max pooling shape: {x.shape}")
    print("CNN COMPLETE......")
    
    
if __name__ == "__main__":
    cnn_drill()