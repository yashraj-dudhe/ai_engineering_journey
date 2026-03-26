import time
import random
from typing import List, Optional

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"function '{func.__name__}' took {end_time - start_time:.4f} seconds")
        return result
    return wrapper 


class DataProcessor:
    def __init__(self, data_size:int =1000):
        self.data_size = data_size
        self.data:List[int] = []
    
    @timer_decorator
    def generate_data(self,min_value: int,max_value: int):
        print(f"generating {self.data_size} records ...")
        self.data = [random.randint(min_value,max_value) for _ in range(self.data_size)]
        
        
    @timer_decorator
    def process_data(self, threshold: int = 50) -> List[int]:
        filtered_data = [x for x in self.data if x> threshold]
        print(f"filtered down to {len(filtered_data)} records")
        return filtered_data
    
    
    
if __name__ == "__main__":
    processor = DataProcessor(data_size = 1_000_000)
    processor.generate_data(min_value = 5000, max_value=10000)
    result = processor.process_data(threshold=75)
    print("job complete")