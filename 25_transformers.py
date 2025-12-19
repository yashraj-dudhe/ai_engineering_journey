import torch
import torch.nn as nn

def transformer_drill():
    print("The setup----")
    batch_size = 1
    seq_len = 10
    embed_dim = 512
    
    
    inputs = torch.randn(batch_size,seq_len,embed_dim)
    print(f"input shape:{inputs.shape}")
    
    print("Transformer layer--")
    encode_layer = nn.TransformerEncoderLayer(d_model = embed_dim,nhead = 8,batch_first= True)
    output = encode_layer(inputs)
    
    print(f"output shape: {output.shape}")
    
    print("The inspection")
    
    if inputs.shape == output.shape:
        print("shape preservered")
    else:
        print("shape mismatch")
        
    
    
if __name__ == "__main__":
    transformer_drill()