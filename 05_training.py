import numpy as np
from typing import List, Tuple

class TrainableNeuron:
    def __init__(self, input_size: int):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = 0.1
        print(f"born stupid weights:{self.weights}, bias:{self.bias:.2f}")
        
        
    def activation(self, x: float) -> int:
        return 1 if x> 0 else 0
    
    
    def predict(self, input: np.ndarray) -> int:
        z = np.dot(input,self.weights) + self.bias
        return self.activation(z)
    
    def train(self, X : List[List[int]], y: List[int], epochs: int = 100):
        print(f"Traing started for {epochs} epochs")
        
        for epoch in range(epochs):
            total_error = 0
            
            for i in range(len(X)):
                input = np.array(X[i])
                target = y[i]
                
                prediction = self.predict(input)
                error = target - prediction
                
                self.weights += error*input*self.learning_rate
                self.bias += error*self.learning_rate
                
                total_error += abs(error)
                
            if epoch%5 == 0 or total_error == 0:
                print(f"epochs: {epoch} total error: {total_error}")
                
            if total_error == 0:
                print("I have learned the patter")
                break
        
        
if __name__ == "__main__":
    X_train = [[0,0],[0,1],[1,0],[1,1]]
    y_train = [0,1,1,0]
    
    neuron = TrainableNeuron(input_size=2)
    neuron.train(X_train,y_train)
    
    print("final results....")
    print(f"\n weights: {neuron.weights}")
    print(f"\n bias: {neuron.bias}")
    
    for input in X_train:
        print(f"inputs:{input} output: {neuron.predict(np.array(input))}")