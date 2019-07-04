import sys
import os
from PIL import Image
import numpy as np

from mnist import load_mnist

# 부모 디렉토리도 사용
sys.path.append(os.pardir)

def imshow(I):
    '''
    mnist 이미지를 표시해주는 함수
    :I {np.array} : mnist training or test data
    :return {void} : there is no return value
    '''
    img = Image.fromarray(np.uint8(I))
    img.show()

# 교차 엔트로피 손실함수
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    cross_batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / cross_batch_size

def gradient(f, x):
    h = 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]
        fxph = f(tmp + h)
        fxmh = f(tmp - h)
        grad[idx] = (fxph - fxmh) / (2 * h)

    return grad

# mnist 이미지 가져오기
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# 데이터 차원 살펴보기
# print(x_train.shape)
# print(x_test.shape)
# print(t_train.shape)
# print(t_test.shape)


# 데이터 하나 확인해보기
# I = x_train[0]
# I = I.reshape(28, 28)
# imshow(I)


train_size = x_train.shape[0]
batch_size = 10
mini_batch_index = np.random.choice(train_size, batch_size)
print(mini_batch_index)
x_batch = x_train[mini_batch_index]
t_batch = t_train[mini_batch_index]
print(t_batch.shape)
