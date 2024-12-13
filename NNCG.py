import numpy as np 

class Tensor():
    def __init__(self, data):
        self.data = data
        self.grad = 0.0

    def __repr__(self):
        return f"Tensor({self.data})"
    
    def __add__(self, other):
        return Tensor(self.data + other.data)
    
    def __mul__(self, other):
        return Tensor(self.data * other.data)
    
if __name__ == "__main__":
    t1 = Tensor(3.0)
    t2 = Tensor(4.0)
    print(t1 + t2, t1 * t2)