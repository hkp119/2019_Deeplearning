import numpy as np
from dataset.mnist import load_mnist
from twolayernets import TwolayerNet

(x_tr, t_tr), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

input_size = x_tr.shape[1]
hidden_size = 100
output_size = 10

net = TwolayerNet(input_size, hidden_size, output_size)

iters_num = 10000
train_size = x_tr.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 1

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_tr[batch_mask]
    t_batch = t_tr[batch_mask]

    grad = net.gradient(x_batch, t_batch)

    for key in('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * grad[key]

    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        print("epoch count ", epoch_cnt)
        train_acc = net.accuracy(x_tr, t_tr)
        test_acc = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(train_acc, test_acc)
        epoch_cnt += 1

print("훈련 완료")
