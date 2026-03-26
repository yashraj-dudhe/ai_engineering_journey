import torch
import torch.nn as nn
import torch.optim as optim

text = "i like ai i like code code is fun"
words = text.split()
vocab = list(set(words))
word_to_idx = {w: i for i,w in enumerate(vocab)}
idx_to_word = {i: w for i,w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"vocab: {word_to_idx}")

input_list = []
target_list = []

for i in range(len(words) - 2):
    input_list.append([word_to_idx[words[i]],word_to_idx[words[i+1]]])
    target_list.append(word_to_idx[words[i+2]])
    
inputs = torch.tensor(input_list)
targets = torch.tensor(target_list)


class GPTLite(nn.Module):
    def __init__(self,vocab_size,embed_dim,nhead,hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim,nhead = nhead,batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,num_layers=1)
        self.fc = nn.Linear(embed_dim,vocab_size)
        
    
    def forward(self, src):
        embeds = self.embedding(src)
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(1))
        
        output = self.transformer_encoder(embeds,mask = mask)
        last_word_output = output[:,-1,:]
        prediction = self.fc(last_word_output)
        return prediction

model = GPTLite(vocab_size,embed_dim=16,nhead=2,hidden_dim=32)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

print("Training GPT Lite")
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs,targets)
    loss.backward()
    optimizer.step()
    
    if epoch%20==0:
        print(f"epoch: {epoch} | Loss: {loss.item():.4f}")
        
print("\n Generating--")
test_seq = torch.tensor([[word_to_idx["code"],word_to_idx["is"]]])

with torch.no_grad():
    prediction_logits = model(test_seq)
    predicted_id = prediction_logits.argmax(dim=1).item()
    print(f"Input:'code is' -> Predicted:{idx_to_word[predicted_id]}")