import torch
import torch.nn as nn


def rnn_anatomy():
    print("1. The Setup....")
    batch_size = 1
    seq_len = 3
    input_size =4
    hidden_size = 2
    
    rnn = nn.RNN(input_size,hidden_size,batch_first=True)
    
    
    print("2.The input...")
    inputs = torch.randn(batch_size,seq_len,input_size)
    print(f"input size:{inputs.shape}")
    
    print("3. Processing")
    hidden_state = torch.zeros(1,batch_size,hidden_size)
    print(f"initial memory:{hidden_state}")
    
    word1 = inputs[:,0:1,:]
    out1, hidden1 = rnn(word1,hidden_state)
    print(f"\n step 1 (after I)\n Memory is now: {hidden1.detach().numpy()}")
    
    word2 = inputs[:,1:2,:]
    out2,hidden2 = rnn(word2,hidden1)
    print(f"\n step 2 (after love)\n memory is now:{hidden2.detach().numpy()}")
    
    word3 = inputs[:,2:3,:]
    out3,hidden3 = rnn(word3,hidden2)
    print(f"\nstep 3 (after AI)\n memory is now:{hidden3.detach().numpy()}")
    
    
    all_out,final_hidden = rnn(inputs,hidden_state)
    
    print(f"final hidden:{final_hidden.detach().numpy()}")
    
    
    
if __name__ == "__main__":
    rnn_anatomy()