import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNSITClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,kernel_size=3,padding = 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(8,32,kernel_size=3,padding = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*7*7, 10)
        
        
    def forward(self,x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc1(x)
        
        
def train_job():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on: {device}")
    
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root = './data', train = True, download = True,transform = transform)
    
    loader = DataLoader(train_data,batch_size = 64,shuffle = True)
    
    
    model = MNSITClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Started ......")
    
    for epoch in range(3):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch_idx, (data,targets) in enumerate(loader):
            data,targets = data.to(device),targets.to(device)
            
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss+=loss.item()
            
            predictions = output.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            
            if batch_idx % 100 == 0:
                print(f"epoch: {epoch}| batch_idx: {batch_idx}| loss: {loss.item():.4f}")
            
    
        accuracy = 100 * correct/total_samples
        print(f"{epoch} epoch complete. Average loss: {total_loss/len(loader):.4f} | Accuracy: {accuracy:.2f}%")
        
        
    torch.save(model.state_dict(), "mnist_cnn.path")
    print("\n model saved to 'mnist_cnn.path")    
            

if __name__ == "__main__":
    train_job()
