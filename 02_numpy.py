import time
import numpy as np
import random


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args,**kwargs)
        print(f"{func.__name__} took {time.time()- start:.5f} seconds")
        return res
    return wrapper

@timer
def python_list_method(size: int):
    data = [random.random() for _ in range(size)]
    result = [x*5 for x in data]
    return result


@timer
def numpy_array_method(size: int):
    data = np.random.rand(size)
    result = data*5
    return result


if __name__ == "__main__":
    N = 10_000_000
    print(f"race starting with {N} elements")
    python_list_method(N)
    numpy_array_method(N)
    