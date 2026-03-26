import torch

def tensor_drill():
    print("1. creating the data")
    
    x= torch.arange(24)
    print(f"original: {x}")
    print(f"shape: {x.shape}")
    

    print("2. reshaping")
    
    matrix = x.view(4,6)
    print(f"Matrix: \n {matrix}")
    print(f"shape: {matrix.shape}")    
    
    print("3. image shape")
    
    image = x.view(2,3,4)
    print(f"image batch:\n {image}")
    print(f"image shape: {image.shape}")
    
    
    print("4. permuting")
    
    swapped = image.permute(1,2,0)
    print(f"swapped shape: {swapped.shape}")
    
    print("everything complete")
    
    
if __name__ == "__main__":
    tensor_drill()