import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def data_drill():
    print("1. The Transfrom setup......")
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    
    print("2. downloading MNIST.....")
    train_data = datasets.MNIST(root='./data', train = True, download = True,transform = transform)
    print(f"total images: {len(train_data)}")
    
    
    print("3. The data loader .....")
    train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
    
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    print(f"Batch shape:{images.shape}")
    print(f"Labels shape:{labels.shape}")
    
    print("4. The visualization......")
    plt.imshow(images[0].squeeze(), cmap = "gray")
    plt.title(f"Label: {labels[0].item()}")
    plt.show()
    
    
if __name__ == "__main__":
    data_drill()