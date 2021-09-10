import numpy as np
from numpy import ndarray

def square(x: ndarray) -> ndarray:
    '''
    인자로 받은 ndarray 배열의 각 요솟값을 제곱한다.
    '''
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    '''
    ndarray 배열의 각 요소에 'Leaky ReLU' 함수를 적용한다.
    '''
    return np.maximum(0.2 * x, x)

from typing import Callable

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    '''
    배열 input의 각 요소에 대해 함수 func의 도함숫값 계산
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

from typing import List

# ndarray를 인자로 받고 ndarray를 반환하는 함수
Array_Function = Callable[[ndarray], ndarray]

# Chain은 함수의 리스트다.
Chain = List[Array_Function]

def chain_length_2(chain: Chain,
                   x: ndarray) -> ndarray:
    '''
    두 함수를 연쇄(chain)적으로 평가
    '''

    assert len(chain) == 2, "인자 chain의 길이는 2여야 함"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))
    
def sigmoid(x: ndarray) -> ndarray:
    '''
    입력으로 받은 ndarray의 각 요소에 대한 sigmoid 함숫값을 계산한다.
    '''
    return 1 / (1 + np.exp(-x))

def chain_deriv_2(chain: Chain,
                  input_range: ndarray) -> ndarray:
    '''
    두 함수로 구성된 합성함수의 도함수를 계산하기 위해 연쇄법칙을 사용함
    (f2(f1(x))' = f2'(f1(x)) * f1'(x)
    '''

    assert len(chain) == 2, "인자 chain의 길이는 2여야 함"
    assert input_range.ndim == 1, "input_range는 1차원 ndarray여야 함"

    f1 = chain[0]
    f2 = chain[1]

    # f1(x)
    f1_of_x = f1(input_range)

    # df1/du
    df1dx = deriv(f1, input_range)
    
    #df2/du(f1(x))
    df2du = deriv(f2, f1(input_range))

    return df1dx * df2du

# PLOT_RANGE = np.arange(-3, 3, 0.01)

# chain_1 = [square, sigmoid]
# chain_2 = [sigmoid, square]

# plot_chain(chain_1, PLOT_RANGE)
# plot_chain_deriv(chain_1, PLOT_RANGE)

def chain_deriv_3(chain: Chain,
                  input_range: ndarray) -> ndarray:
    '''
    두 함수로 구성된 합성함수의 도함수를 계산하기 위해 연쇄법칙을 사용함
    (f3(f2(f1(x))))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)
    '''

    assert len(chain) == 3, "인자 chain의 길이는 3이어야 함"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    # f1(x)
    f1_of_x = f1(input_range)

    # f2(f1(x))
    f2_of_x = f2(f1_of_x)

    # df3/du
    df3du = deriv(f3, f2_of_x)

    # df2/du
    df2du = deriv(f1, f1_of_x)
    
    #df1dx
    df1dx = deriv(f1, input_range)

    return df1dx * df2du * df3du
