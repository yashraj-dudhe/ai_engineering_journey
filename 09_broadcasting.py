import torch

def broadcast_drill():
    print("1. same shape math")
    a = torch.tensor([1,2,3])
    b = torch.tensor([4,5,6])
    
    print(f"a+b: {a + b}")
    
    print("2. broadcasting")
    scalar = torch.tensor(10)
    print(f"a+10:{a+scalar}")
    
    print("3. matrix broadcasting")
    matrix = torch.tensor([[1,2,3],
                            [4,5,6]])
    
    row_vec = torch.tensor([10,10,30])
    
    result = matrix + row_vec
    print(f"matrix + vector:\n {result}")
    

if __name__ == "__main__":
    broadcast_drill()