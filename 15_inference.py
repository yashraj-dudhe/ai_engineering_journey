import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


from trainmnist import MNSITClassifier

def predict_custom_image():
    print("---waking up the model----")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MNSITClassifier().to(device)
    
    model.load_state_dict(torch.load("mnist_cnn.path",map_location = device))
    model.eval()
    print("---model loaded and ready----")
    
    print("--preparing fake input--")
    dummy_input = torch.randn(1,1,28,28).to(device)
    
    print("---the prediction---")
    with torch.no_grad():
        outputs = model(dummy_input)
        
        probabilities = torch.softmax(outputs,dim=1)
        
        prediction = outputs.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()*100
        
        print(f"i predict this random noise is: {prediction}")
        print(f"confidence:{confidence:.2f}")
        
        
if __name__ == "__main__":
    predict_custom_image()