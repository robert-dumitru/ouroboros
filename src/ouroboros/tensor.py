from __future__ import annotations
import numpy as np

_T = int | float | list | np.ndarray


class Tensor:

    def __init__(self, data: np.ndarray, _children: tuple[Tensor, ...] = (), _op: str = "") -> None:
        self.data: np.ndarray = data
        self.grad: np.ndarray = np.zeros(data.shape)
        # internal variables
        self._backward: callable = lambda: None
        self._prev: set[Tensor, ...] = set(_children)
        self._op: str = _op

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    @classmethod
    def as_tensor(cls, data: Tensor | _T) -> Tensor:
        if isinstance(data, Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return cls(data)
        elif isinstance(data, list):
            return cls(np.array(data))
        elif isinstance(data, (int, float,)):
            return cls(np.array(data))
        else:
            raise NotImplementedError

    @property
    def T(self) -> Tensor:  # noqa: E
        out: Tensor = Tensor(self.data.T, self.grad.T)
        out._backward = self._backward
        out._prev = self._prev
        out._op = self._op
        return out

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __add__(self, other: Tensor | _T) -> Tensor:
        other: Tensor = Tensor.as_tensor(other)
        out: Tensor = Tensor(self.data + other.data, (self, other,), "+")

        def _backward():
            self.grad += out.grad
            other.grad += other.grad
        out._backward = _backward

        return out

    def __radd__(self, other: Tensor):
        return self + other

    def __mul__(self, other: Tensor | _T) -> Tensor:
        other: Tensor = Tensor.as_tensor(other)
        out: Tensor = Tensor(self.data * other.data, (self, other,), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other: Tensor):
        return self * other

    def __pow__(self, other: int | float) -> Tensor:
        assert isinstance(other, (int, float,))
        out = Tensor(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * np.full(self.data.shape, -1)

    def __sub__(self, other: Tensor | _T):
        return self + (-other)

    def __rsub__(self, other: Tensor):
        return other + (-self)

    def __truediv__(self, other: Tensor | _T):
        return self * other**-1

    def __rtruediv__(self, other: Tensor):
        return other * self**-1

    def __matmul__(self, other: Tensor | _T):
        other: Tensor = Tensor.as_tensor(other)
        out: Tensor = Tensor(self.data @ other.data, (self, other,), "@")

        def _backward():
            if len(out.grad.shape) == 0:
                self.grad += out.grad * other.data
                other.grad += self.data * out.grad
            else:
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def __rmatmul__(self, other: Tensor):
        return (other.T @ self.T).T

    def backward(self):

        topo: list[Tensor] = []
        visited: set[Tensor, ...] = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:  # noqa: E
                    build_topo(c)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones(self.grad.shape)
        for v in reversed(topo):
            v._backward()
