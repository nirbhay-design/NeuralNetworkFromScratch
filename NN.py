import numpy as np 
from utils import *

class Linear():
    def __init__(self, in_features, out_features):
        self.w = np.random.uniform(-1/in_features, 1/in_features, size=(out_features, in_features)) 
        self.b = np.random.uniform(-1/out_features, 1/out_features, size=(out_features, 1))

        self.gradw = None 
        self.gradb = None 

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        # x.shape: [B, in_features]
        self.x = x.T 
        lin_out = self.w @ self.x + self.b
        return lin_out.T

    def backward(self, grad):
        # grad.shape: [B, out_features]
        grad = grad.T
        self.gradb = np.sum(grad, axis=1).reshape(-1,1)
        self.gradw = grad @ self.x.T
        self.gradx = self.w.T @ grad
        return self.gradx.T
    
class MLP():
    def __init__(self, network):
        self.network = network 

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = x 
        for layer in self.network:
            out = layer(out)
        return out
    
    def backward(self, grad):
        rungrad = grad 
        for layer in self.network[::-1]:
            rungrad = layer.backward(rungrad)
    
    def sgd(self, lr):
        for idx, layer in enumerate(self.network):
            if isinstance(layer, Linear):
                self.network[idx].w -= lr * layer.gradw
                self.network[idx].b -= lr * layer.gradb

class Trainer():
    def __init__(self, mlp, loss):
        self.mlp = mlp 
        self.loss = loss

    def train(self, x, y, epochs = 10, lr = 0.01):
        for i in range(epochs):
            y_pred = self.mlp(x)
            loss = self.loss(y, y_pred)

            loss_grad = self.loss.backward()
            self.mlp.backward(loss_grad)
            self.mlp.sgd(lr)

            print(f"epoch: {i+1}, loss: {loss:.3f}")


if __name__ == "__main__":
    B = 10; in_feat = 4; cls = 3
    x = np.random.rand(B,in_feat) # sample data
    y = np.array([np.random.randint(0, cls) for _ in range(B)]) # sample features
    network = [
        Linear(in_feat,5),
        Sigmoid(),
        Linear(5,5),
        Tanh(),
        Linear(5,6),
        SiLU(),
        Linear(6,6),
        ReLU(),
        Linear(6,cls),
        Softmax()
    ]

    mlp = MLP(network)
    loss = CrossEntropyLoss()
    trainer = Trainer(mlp, loss)
    trainer.train(x, y, epochs=100)
    

    



