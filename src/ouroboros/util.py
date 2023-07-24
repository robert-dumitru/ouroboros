import numpy as np
from .tensor import Tensor


def gradient(func: callable, value: Tensor | np.ndarray) -> np.ndarray:
    in_tensor = Tensor.as_tensor(value)
    out: Tensor = func(in_tensor)
    out.backward()
    return in_tensor.grad


def hessian(func: callable, value: Tensor | np.ndarray) -> np.ndarray:
    raise NotImplementedError
