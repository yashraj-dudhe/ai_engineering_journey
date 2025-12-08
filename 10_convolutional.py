import torch
import torch.nn.functional as F

def vision_drill():
    print("1. The image 6*6 image drill")
    
    image = torch.tensor([
        [10,10,10,0,0,0],
        [10,10,10,0,0,0],
        [10,10,10,0,0,0],
        [10,10,10,0,0,0],
        [10,10,10,0,0,0],
        [10,10,10,0,0,0]
    ]).float()
    
    
    image = image.view(1,1,6,6)
    print(f"image shape: {image.shape}")
    
    
    print("2. The filter (vertical edge detector)")
    
    filter_weights = torch.tensor([
        [1,0,-1],
        [1,0,-1],
        [1,0,-1]
    ]).float()
    
    
    filter_weights = filter_weights.view(1,1,3,3)
    
    print("3. the convolution scanning")
    
    output = F.conv2d(image,filter_weights)
    print(f"output shape: {output.shape}")
    print("output feature map")
    print(output.squeeze())
    
    
if __name__ == "__main__":
    vision_drill()