import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

from cifar import CifarNet

def evaluate_model():
    device = torch.device("cpu")
    print("1 Loading Data ----")
    stats = ((0.5,0.5,0.5),(0.5,0.5,0.5))
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    test_data = datasets.CIFAR10(root = "./data",train = False,download = True, transform = test_transform)
    test_loader = DataLoader(test_data,batch_size = 64,shuffle = False)
    
    print("2. Loading the model----")
    model = CifarNet().to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("3. Running Predictions -----")
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("4. Visualizing----")
    classes = test_data.classes
    
    cm = confusion_matrix(all_labels,all_preds)
    
    plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True,fmt='d',xticklabels=classes,yticklabels=classes,cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('CIFAR-10 Confusion Matrix')
    plt.show()
    
    
    
if __name__ == "__main__":
    evaluate_model()