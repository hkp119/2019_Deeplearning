import sys, os
from mnist import load_mnist
import numpy as np
from PIL import Image

# 부모 디렉토리도 사용
sys.path.append(os.pardir)

# 이미지 디스플레이 함수 정의
def imshow(I):
    img = Image.fromarray(np.uint8(I))
    img.show()

# mnist 이미지 가져오기
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

# 데이터 차원 살펴보기
print(x_train.shape)
print(x_test.shape)
print(t_train.shape)
print(t_test.shape)

# 데이터 하나 확인해보기
I = x_train[0]
I = I.reshape(28, 28)
imshow(I)