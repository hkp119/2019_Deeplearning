import numpy as np

def softmax(z):
    '''
    소프트맥스 함수
    : z {np.array} : 입력 인수
    : return {np.array} : 출력 인수
    '''
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
    return -np.sum(t*np.log(y + delta))
   
