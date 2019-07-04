import numpy as np

def gradient(f, x):
    '''
    수치적 그래디언트를 구해주는 함수
    : f {function} : target function
    : x {np.array} : target parameter
    : return {np.array} : gradient vector
    '''
    h = 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad

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
