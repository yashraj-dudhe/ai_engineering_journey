import torch
import torch.nn as nn


def embedding_drill():
    print("1.The setup---")
    vocab_size = 10
    embed_dim = 4
    
    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim = embed_dim)
    print(f"embedding layer created. Table shape:{embedding_layer.weight.shape}")
    
    print("2 The lookup---")
    input_ids = torch.tensor([2,3,4])
    
    vectors = embedding_layer(input_ids)
    
    print(f"input ids: {input_ids}")
    print(f"output vectors:\n{vectors}")
    print(f"output shape:{vectors.shape}")
    
    print("The concept check----")
    vec_I = vectors[0]
    vec_love = vectors[1]
    
    
    cos = nn.CosineSimilarity(dim=0)
    similarity = cos(vec_I,vec_love)
    
    print(f"similarity between 'I' and 'Love' (random):{similarity.item():.4f}")
    

if __name__ == "__main__":
    embedding_drill()