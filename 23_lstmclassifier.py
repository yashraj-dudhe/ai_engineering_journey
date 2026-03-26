import torch
import torch.nn as nn
import torch.optim as optim

data = [
    ("I love this movie",1),
    ("this is great",1),
    ("fantastic story",1),
    ("I hate this movie",0),
    ("this is bad",0),
    ("i do not love this",0)
]

word_to_idx = {"<PAD>":0}
for sentence,label in data:
    for word in sentence.split():
        if word not in word_to_idx:
            word_to_idx[word]=len(word_to_idx)
            


class LSTMClassifier(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        
        self.lstm = nn.LSTM(embed_dim,hidden_dim,batch_first=True)
        
        self.fc = nn.Linear(hidden_dim,1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self,x):
        embeds = self.embedding(x)
        lstm_out, (hidden,cell) = self.lstm(embeds)
        final_memory = hidden[-1]
        output = self.fc(final_memory)
        return self.sigmoid(output)
    
    
def make_tensor(sentence,word_to_idx):
    ids = [word_to_idx[w] for w in sentence.split()]
    return torch.tensor(ids,dtype=torch.long).unsqueeze(0)


def train_lstm():
    model = LSTMClassifier(vocab_size=len(word_to_idx),embed_dim = 10,hidden_dim = 16)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr =0.01)
    
    print("Training LSTM")
    for epoch in range(100):
        total_loss = 0
        for sentence,label in data:
            model.zero_grad()
            inputs = make_tensor(sentence,word_to_idx)
            target = torch.tensor([label],dtype = torch.float).unsqueeze(0)
            
            output = model(inputs)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            
        if epoch%20 ==0:
            print(f"epcoh:{epoch}|Loss:{total_loss:.4f}")
            
            
    print("test case")
    test_sen = "I do not love this"
    with torch.no_grad():
        inputs = make_tensor(test_sen,word_to_idx)
        prob = model(inputs).item()
        print(f"sentence:'{test_sen}'")
        print(f"probabitlity:{prob:.4f}")
        print(f"Result:{'POSITIVE' if prob>0.5 else 'NEGATIVE'}")
        
        
if __name__ == "__main__":
    train_lstm()
                