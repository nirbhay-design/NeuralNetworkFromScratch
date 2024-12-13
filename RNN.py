import numpy as np 
from utils import *

class RNN():
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        self.A = np.random.uniform(-1/hidden_dim, 1/hidden_dim, size = (hidden_dim, hidden_dim))
        self.B = np.random.uniform(-1/embed_dim, 1/ embed_dim, size = (hidden_dim, embed_dim))
        self.C = np.random.uniform(-1/hidden_dim, 1/hidden_dim, size = (vocab_size, hidden_dim))
        self.ba = np.random.uniform(-1/hidden_dim, 1/hidden_dim, size = (hidden_dim, 1))
        self.bc = np.random.uniform(-1/vocab_size, 1/vocab_size, size = (vocab_size,1))

        self.hidden_dim = hidden_dim

        self.dA = None 
        self.dB = None
        self.dC = None 
        self.dba = None 
        self.dbc = None 
        
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # [B, seqlen, embed_dim]
        x = x.transpose((1,0,2)) # [seqlen, B, embed_dim]
        seqlen, B, _ = x.shape

        self.ht = {-1: np.zeros((self.hidden_dim, B)).T}
        self.ot = {}

        for i in range(seqlen):
            ht = tanh(self.A @ self.ht[i-1].T + self.B @ x[i].T + self.ba)
            ot = (self.C @ ht + self.bc).T

            self.ht[i] = ht.T 
            self.ot[i] = ot 

        return self.ht, self.ot
    
    def backward(self, x):
        pass 

if __name__ == "__main__":
    rnn = RNN(5, 7, 24)
    a = np.random.rand(2, 10, 5)
    h, o = rnn(a)

    print(h[5].shape)
    print(o[8].shape)