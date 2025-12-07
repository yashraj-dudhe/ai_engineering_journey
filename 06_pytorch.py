import torch
import numpy as np

def verifytorch():
    print(f"pytroch version: {torch.__version__}")
    
    x = torch.tensor([[1,2],[3,4]])
    print(f"created tensor:\n {x}")
    
    
    y = x*10
    print(f"performed multiplication operation and result:\n {y}")
    
    numpyarray = np.array([5,6,7])
    tensor_from_numpy = torch.from_numpy(numpyarray)
    print(f"numpy converted to tensor:\n {tensor_from_numpy}")
    
    
if __name__ == "__main__":
    verifytorch()