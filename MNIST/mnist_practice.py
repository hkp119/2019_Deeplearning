import sys, os
from mnist import load_mnist
import numpy as np
from PIL import Image

sys.path.append(os.pardir)

def imshow(I):
    img = Image.fromarray(np.uint8(I))
    img.show()

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(x_test.shape)
print(t_train.shape)
print(t_test.shape)

I = x_train[0]
I = I.reshape(28, 28)
imshow(I)