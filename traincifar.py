import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


from cifar import CifarNet


def train_cifar():
    device = torch.device("cpu")
    
    print("1. Data Augmentation----")
    
    stats = ((0.5,0.5,0.5), (0.5,0.5,0.5))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    print("2. Download cifar 10---")
    train_data = datasets.CIFAR10(root = './data',train = True,download = True,transform = train_transform)
    test_data = datasets.CIFAR10(root = './data',train = True,download = True,transform = test_transform)    
    
    train_loader = DataLoader(train_data,batch_size = 32,shuffle = True)
    test_loader = DataLoader(test_data,batch_size = 32,shuffle = False)
    
    print("3. Initialize model")
    model = CifarNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    
    print("Training Started 1 epoch only due to cpu---")
    model.train()
    total_loss = 0
    
    for i,(images,labels) in enumerate(train_loader):
        images,labels = images.to(device),labels.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i%50==0:
            print(f"Batch {i}/{len(train_loader)}|Loss: {loss.item():.4f}")
            
        if i>2000:
            print("early stopping for cpu sanity check")
            break
        
    print("training loop verfied")
        
if __name__ == "__main__":
    train_cifar()
