import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]]).float()
y = torch.tensor([[0],[1],[1],[0]]).float()

class xormodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2,10)
        self.activation = nn.ReLU()
        self.output = nn.Linear(10,1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self,x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
    
    
model = xormodel()

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr = 0.1)

print("training xor model")

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    
    loss = criterion(output,y)
    
    loss.backward()
    optimizer.step()
    
    if epoch%100 == 0:
        print(f"epoch: {epoch} error:{loss.item():.4f}")
        

print("The final predictions")
with torch.no_grad():
    predictions = model(X)
    rounded = predictions.round()
    print(f"inputs:\n{X}")
    print(f"predictions:\n {rounded}")
    print(f"target:\n {y}")