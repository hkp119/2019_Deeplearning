import numpy as np
from mnist import load_mnist
from TwoLayerNet import TwolayerNet

(x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

iter_num = 1000
train_size = x_train.shape[0]
input_size = x_train.shape[1]
hidden_size = 100
output_size = 10
batch_size = 100
learning_rate = 0.1

net = TwolayerNet(input_size=input_size, \
        hidden_size=hidden_size, output_size=output_size)

while iter_num != 0:
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = net.numerical_gradient(x_batch, t_batch)
    print(grad)

    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * grad[key]
    
    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print(loss)
    
    iter_num -= 1
