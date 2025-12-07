import numpy as np 
from typing import List

class ArtificialNeuron:
    def __init__(self, weights: List[float], bias: float):
        self.weights = np.array(weights)
        self.bias = bias
        print(f"nueron created with weights{self.weights} and bias{self.bias}")
    
    def ActivationFunction(self, x: float) -> int:
        return 1 if x>0 else 0
    
    def forward(self, inputs: List[float]) -> int:
        x = np.array(inputs)
        z = np.dot(x, self.weights) + self.bias
        
        output = self.ActivationFunction(z)
        
        print(f"input: {x} weighted sum:{z} output: {output}")
        
        return output
    
if __name__ == "__main__":
    neuron = ArtificialNeuron(weights = [0.5,0.5],bias = -0.7)
    
    print("\n Testing the neuron......")
    test_data = [
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]
    
    for x in test_data:
        neuron.forward(x)