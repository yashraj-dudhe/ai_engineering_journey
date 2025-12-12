import torch

def tokenizer_drill():
    print("1. The corpus (data)---")
    
    sentences = [
        "I love AI",
        "I love coding",
        "Python is great",
        "Deep learning is hard but fun"
    ]
    print(f"raw data:{sentences}")
    
    
    print("2. Tokenization ---")
    
    tokenized_sent = [s.lower().split() for s in sentences]
    print(f"Tokenized: {tokenized_sent[0]}")
    
    print("3. Building the vocabulary")
    vocab = {"<PAD>":0,"<UNK>":1}
    
    idx = 2
    for sentence in tokenized_sent:
        for word in sentence:
            if word not in vocab:
                vocab[word] = idx
                idx+=1

    print(f"vocabulary size: {len(vocab)}")
    print(f"Dictionary: {vocab}")
    
    
    print("3. Encoding text -> numbers")
    input_text = "I love Java"
    tokens = input_text.lower().split()
    
    ids = [vocab.get(token,1) for token in tokens]
    
    tensor_input = torch.tensor(ids)
    print(f"Input:{input_text}")
    print(f"Tensor: {tensor_input}")
    
    
if __name__ == "__main__":
    tokenizer_drill()