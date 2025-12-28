import torch
import torch.nn.functional as F
import math

def manual_attention_drill():
    print("The setup 3 words-------")
    #3 words: "Bank", "Money", "River"
    
    Q = torch.tensor([[[1.0,0.0,1.0,0.0],
                       [0.0,1.0,0.0,1.0],
                       [1.0,1.0,1.0,1.0]]])
    
    K = torch.tensor([[[1.0,0.0,1.0,0.0],
                       [0.0,10.0,0.0,10.0],
                       [1.0,1.0,1.0,1.0]]])
    
    V = torch.tensor([[[10.0,0.0,0.0,0.0],
                       [0.0,100.0,0.0,0.0],
                       [1.0,1.0,1000.0,0.0]]])
    
    d_k = 4
    
    print("\n step 1: the matmul q*k transpose----")
    scores = torch.matmul(Q,K.transpose(-2,-1))
    
    print(f"Raw scores: \n{scores}")
    
    print("step 3: the scale----")
    
    scores =  scores/math.sqrt(d_k)
    print(f"scaled scores:\n{scores}")
    
    print("step 4 : softmax---")
    attention_weights = F.softmax(scores,dim=-1)
    print(f"attention weights:\n{attention_weights}")
    
    print("step 5: weighted value")
    output = torch.matmul(attention_weights, V)
    
    print(f"final context output:\n{output}")
    
    
    
if __name__ == "__main__":
    manual_attention_drill()
    