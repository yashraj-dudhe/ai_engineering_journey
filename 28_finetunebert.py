import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW 
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class TinyReviewDataset(Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)


def train_bert():
    print("1. Preparing data....")
    texts = [
        "I absolutely love this product","Best purchase ever","Fantastic quality",
        "It was ok but not great","Terrible experience","I hate this",
        "Waste of money","Do not buy this"
    ]

    labels = [1,1,1,0,0,0,0,0]
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    encodings = tokenizer(texts,truncation = True, padding = True, max_length = 16)
    
    dataset = TinyReviewDataset(encodings,labels)
    loader = DataLoader(dataset, batch_size=2,shuffle = True)
    
    
    print("2. Loading pretrained model....")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels = 2)
    model.train()
    
    optimizer = AdamW(model.parameters(),lr=1e-5)
    
    print("3. Finetuning loop....")
    
    for epoch in range(10):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"epoch: {epoch}| loss:{total_loss:.4f}")
    
    print("4 Test time...")
    model.eval()
    test_text = "this is the worst thing i have ever brought"
    inputs = tokenizer(test_text,return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits,dim=1).item()
        
    print(f"Input: {test_text}")
    print(f"prediction:{'Positive' if prediction==1 else 'Negative'}")
    

if __name__ == "__main__":
    train_bert()