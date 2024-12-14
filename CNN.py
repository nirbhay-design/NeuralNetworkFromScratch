import numpy as np 

class Conv():
    def __init__(self, in_channel, out_channel, kernel_size):
        self.ic = in_channel 
        self.oc = out_channel
        self.ks = kernel_size 
        self.kernel = np.random.uniform(-1/in_channel, 1/in_channel, size=(out_channel, in_channel, kernel_size, kernel_size))
        self.b = np.random.uniform(-1/in_channel, 1/in_channel, size=(out_channel, 1))

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return self.convolution(x)
    
    def convolution(self, x):
        # [B, C, H, W]
        B, C, H, W = x.shape 
        Hn = H - self.ks + 1
        Wn = W - self.ks + 1
        out = np.zeros((B, self.oc, Hn, Wn))
        for i in range(self.oc):
            ck = self.kernel[i, ...][np.newaxis, :, :, :].repeat(repeats = B, axis=0) # [B, in_channel, kernel_size, kernel_size]
            for h in range(Hn):
                for w in range(Wn):
                    out[:, i, h, w] = np.sum(ck * x[:, :, h:h+self.ks, w:w+self.ks])
        return out 
    
    def backward(self, grad):
        pass 

if __name__ == "__main__":
    conv = Conv(3, 4, 3)
    img = np.random.rand(1,3,5,5)
    out = conv(img)
    print(out)

    
