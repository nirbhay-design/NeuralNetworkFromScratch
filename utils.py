import numpy as np 

class Sigmoid():
    def __init__(self):
        self.s = lambda x: 1 / (1 + np.exp(-x))
        self.sd = lambda y: y * (1 - y)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.y = self.s(x)
        return self.y

    def backward(self, grad):
        return grad * self.sd(self.y)

class Tanh():
    def __init__(self):
        self.t = lambda x: (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))
        self.td = lambda y: 1 - y**2

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.y = self.t(x)
        return self.y 

    def backward(self, grad):
        return grad * self.td(self.y)
    
class ReLU():
    def __init__(self):
        self.r = lambda x: x * (x > 0)
        self.rd = lambda x: (x > 0).astype(np.float64)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.y = self.r(x)
        return self.y

    def backward(self, grad):
        return grad * self.rd(self.y)
    
class SiLU():
    def __init__(self):
        self.sg = lambda x: 1 / (1 + np.exp(-x))
        self.s = lambda x: (x * self.sg(x), x, self.sg(x))
        self.sd = lambda x, sx: sx * (1 + x * (1 - sx)) 

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.y, self.x, self.sgx = self.s(x)
        return self.y        

    def backward(self, grad):
        return grad * self.sd(self.x, self.sgx)
    
class Softmax():
    def __init__(self):
        self.s = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1,1)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # X: [B, F]
        self.y = self.s(x) #[B, F]
        return self.y

    def backward(self, grad):
        B, F = self.y.shape
        M = np.repeat(self.y, F, axis=1).reshape(B, F, F).transpose((0,2,1))
        I = np.repeat(np.eye(F)[np.newaxis, :, :], repeats = B, axis = 0)
        local_grad = M * (I - M.transpose((0,2,1)))
        global_grad = np.matmul(local_grad, grad.reshape(B, F, 1)).reshape(B,F)
        return global_grad
    
class CrossEntropyLoss():
    def __init__(self):
        # y_true one hot, y_pred [B,nc]
        self.loss = lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred + 1e-5))

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

    def forward(self, y_true, y_pred):
        y_true_oh = np.zeros_like(y_pred)
        y_true_oh[np.arange(y_pred.shape[0]), y_true] = 1.0
        self.y_true_oh = y_true_oh
        self.y_pred = y_pred
        return self.loss(self.y_true_oh, self.y_pred)

    def backward(self):
        loss_deri = -1 / (self.y_pred + 1e-5)
        return self.y_true_oh * loss_deri
    
def tanh(x):
    return Tanh()(x)

def relu(x):
    return ReLU()(x)

def silu(x):
    return SiLU()(x)

def sigmoid(x):
    return Sigmoid()(x)

def softmax(x):
    return Softmax()(x)

if __name__ == "__main__":
    pass 