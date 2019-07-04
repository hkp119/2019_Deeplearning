from utils.functions import softmax, cross_entropy_error
from utils.grad import gradient

import numpy as np

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
    
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = SimpleNet()
f = lambda l: net.loss(x, t)

dW = gradient(f, net.W)

print(dW)
    