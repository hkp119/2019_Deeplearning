import sys
import numpy as np

def cost(x):
    '''
    cost function
    f = x0^2 + x1^2
    :x {np.array, dtype = double} : (2, ) 사이즈 
    :return {double} : x0^2 + x1^2 리턴
    '''
    x = x*x
    return np.sum(x)

def gradient(f, x):
    '''
    수치적 그래디언트를 구해주는 함수
    : f {function} : target function
    : x {np.array} : target parameter
    : return {np.array} : gradient vector
    '''
    h = 0.0001
    ret = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]

        x[idx] = tmp + h
        fxph = f(x)

        x[idx] = tmp - h
        fxmh = f(x)

        ret[idx] = (fxph - fxmh) / (2 * h)
        x[idx] = tmp
    return ret 

def gradient_descent(f, init_x, alpha=0.01, step=1000):
    '''
    특정 파라미터에 경사 하강법을 사용하는 함수
    : f {double} : cost function
    : init_x {np.array} : target parameter
    : alpha {double} : learning rate
    : it {int} : number of iteration
    : return {np.array} : learned parameter
    '''
    x = init_x

    while step != 0:
        grad = gradient(f, x)
        x -= alpha * grad
        step -= 1

    return x

init_x = np.array([-3.0, 4.0])
print(init_x)
newx = gradient_descent(cost, init_x=init_x, alpha=0.01, step=1000)
print(newx)
