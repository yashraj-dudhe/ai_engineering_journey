import torch
import torch.nn as nn
import torch.optim as optim

data = [
    ("I love this movie",1),
    ("this is great",1),
    ("fantastic story",1),
    ("I hate this movie",0),
    ("this is bad",0),
    ("terrible story",0),
]

word_to_idx = {"<PAD>":0}
for sentence,label in data:
    for word in sentence.split():
        if word not in word_to_idx:
            word_to_idx[word]=len(word_to_idx)
            
print(f"vocab size: {len(word_to_idx)}")
print(f"vocab: {word_to_idx}")


class TextClassifier(nn.Module):
    def __init__(self,vocab_size,embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.fc = nn.Linear(embed_dim,1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self,x):
        embeds = self.embedding(x)
        sentence_vector = embeds.mean(dim=1)
        output = self.fc(sentence_vector)
        return self.sigmoid(output)
    
    
def make_bow_vector(sentence,word_to_idx):
    ids = [word_to_idx[w] for w in sentence.split()]
    return torch.tensor(ids,dtype = torch.long).unsqueeze(0)

model = TextClassifier(vocab_size = len(word_to_idx), embed_dim = 10)
optimizer = optim.Adam(model.parameters(),lr=0.1)
criterion =  nn.BCELoss()

print("Training")
for epoch in range(100):
    total_loss = 0
    for sentence,label in data:
        model.zero_grad()
        
        inputs = make_bow_vector(sentence,word_to_idx)
        target = torch.tensor([label],dtype = torch.float).unsqueeze(0)
        
        probs = model(inputs)
        
        loss = criterion(probs, target)
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
        
    if epoch % 20 ==0:
        print(f"epoch: {epoch}| loss: {total_loss:.4f}")
        
        
print("Test time")
test_sen = "this movie is not bad"

with torch.no_grad():
    inputs = make_bow_vector(test_sen,word_to_idx)
    prob = model(inputs).item()
    print(f"Sentence:{test_sen}")
    print(f"probability of positive: {prob:.4f}")
    print(f"Sentiment:{'POSITIVE' if prob > 0.5 else 'NEGATIVE'}")
    