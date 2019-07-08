'''
딥러닝에 쓰이는 함수들을 정리해놓은 모듈
넘파이 배열을 인자로 받아 쓰이도록 만듬
밑바닥부터 시작하는 딥러닝 참조
'''
import numpy as np

def softmax(z):
    '''
    소프트맥스 함수
    : z {np.array} : 입력 인수
    : return {np.array} : 출력 인수
    '''
    # 입력 인수가 2차원일때 --> 데이터가 배치단위로 들어올때
    # 도대체 전치는 왜 시키는것인가..
    if z.ndim == 2:
        z = z.T
        z = z - np.max(z, axis=0)
        y = np.exp(z) / np.sum(np.exp(z), axis=0)
        return y.T
    # 입력 인수가 일차원일 때
    C = np.max(z)
    ret = np.exp(z - C)
    sum_val = np.sum(ret)
    ret = ret / sum_val

    return ret

def cross_entropy_error(y, t, delta=1e-7):
    '''
    cross entropy error를 구하는 함수
    : y {np.array} : 추정값
    : t {np.array} : 참값
    : return {double} : 크로스 엔트로피 비용 함수
    '''
    # 배치 형태로 데이터가 들어오지 않을 경우
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    ret = t * np.log(y + delta)
    ret = -np.sum(ret) / batch_size
    return ret
   
def identity_function(x):
    '''
    항등함수 f(x) = x
    : x {np.array} : 입력값
    : return {np.array} : 출력값
    '''
    return x

def step_function(x):
    '''
    계단 함수 f(x) = 0 (x <= 0), f(x) = 1(x > 0)
    : x {np.array} : 입력값
    : return {np.array} : 출력값
    '''
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    '''
    시그모이드 함수 f(x) = 1 / (1 + exp(-x))
    : x {np.array} : 입력값
    : return {np.array} : 출력값
    '''
    return 1 / (1 + np.exp(-x))
    
def sigmoid_diff(x):
    '''
    시그모이드 함수의 도함수를 출력함
    (1 - sigmoid(x))*sigmoid(x)
    : x {np.array} : 입력값
    : return {np.array} : 출력값
    '''
    return (1 - sigmoid(x))*sigmoid(x)

def relu(x):
    '''
    relu 함수를 출력함
    relu(x) = max(0, x)
    : x {np.array} : 입력값
    : return {np.array} : 출력값
    '''
    return np.maximum(0, x)

def relu_diff(x):
    '''
    relu 함수의 도함수를 출력함
    relu'(x) = x < 0에서 0, x >=0에서 1
    : x {np.array} : 입력값
    : return {np.array} : 출력값
    '''
    ret = np.zeros_like(x)
    ret[x >= 0] = 1
    return ret

def MSE(y, t):
    '''
    평균제곱오차를 구하는 함수
    : y {np.array} : 추정치
    : t {np.array} : 레이블
    '''
    return 0.5 * np.sum((y - t)**2)

def softmax_loss(X, t):
    '''
    마지막 출력층에서 소프트맥스함수 + 크로스 엔트로피 함수 동시계산
    : X {np.array} : 활성함수를 거치지 않은 마지막 출력층 결과
    : t {np.array} : 레이블
    '''
    y = softmax(X)
    return cross_entropy_error(y, t)