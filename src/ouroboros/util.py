import numpy as np
from .tensor import Tensor


def gradient(func: callable, value: Tensor | np.ndarray) -> np.ndarray:
    input = Tensor.as_tensor(value)
    out: Tensor = func(input)
    out.backward()
    return input.grad


def hessian(func: callable, value: Tensor | np.ndarray) -> np.ndarray:
    raise NotImplementedError
