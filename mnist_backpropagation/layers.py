import numpy as np
from utils.functions import *


class AffineLayer:
    '''
    행렬의 dot곱의 순전파와 역전파를 구현하는 노드
    : Member variables
        : W {numpy.array} : 가중치 배열
        : b {numpy.array} : 편향 배열
        : x {numpy.array} : 입력 배열
        : dW {numpy.array} : 가중치 배열의 미분
        : db {numpy.array} : 편향 배열의 미분
    : Methods
        : __init__
        : forward
        : backward
    '''
    def __init__(self, W, b):
        '''
        가중치 데이터와 편향 데이터를 인자로 받아 객체를 생성
        : W {numpy.array} : 입력받을 가중치 배열
        : b {numpy.array} : 입력받을 편향 배열
        '''
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        '''
        순전파 메소드
        : x {numpy.array} : 입력 데이터 배열
        : return {numpy.array} : 순전파 연산 결과
        '''
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        ret = np.dot(x, self.W) + self.b
        return ret
    
    def backward(self, dout):
        '''
        역전파 메소드
        : dout {numpy.array} : 역전파로 받은 인자
        : return {numpy.array} : 역전파 연산 결과
        '''
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class ReluLayer:
    '''
    Relu 계층 layer를 구현한 클래스
    : Member Variables
        : out {numpy.array} : 출력값
        : mask {numpy.array} : 해당 원소가 0보다 큰지 작은지 판단하는 마스크
    : Methods
        : __init__ : 초기화 메소드
        : forward : 순전파 메소드
        : backward : 역전파 메소드
    '''
    def __init__(self):
        '''
        초기화 메소드, 멤버 변수 초기화
        '''
        self.out = None
        self.mask = None
    
    def forward(self, x):
        '''
        순전파 메소드
        : x {numpy.array} : 입력 인수
        : out {numpy.array} : 결과를 멤버변수에 저장
        : return {numpy.array} : 순전파 결과
        '''
        self.mask = (x <= 0)
        self.out = x.copy()
        self.out[self.mask] = 0

        return self.out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class SigmoidLayer:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        ret = sigmoid(x)
        self.out = ret 
        return ret 

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx 

class SoftmaxLossLayer:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = softmax_loss(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

