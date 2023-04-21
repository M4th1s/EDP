from typing import Iterable, Any, Callable
import matplotlib.pyplot as plt
from numpy import sin, pi, ndarray, linspace, abs


def graph_func(func: Callable, num:int = 10_000):
    X = linspace(0, 1, num=num)
    Y = [func(x) for x in X]

    plt.plot(X, Y)
    plt.show()


def nsin(X: float | ndarray | Iterable) -> Any:
    return sin(2 * pi * X)


def carre(X: float | ndarray):
    return (X - 1/2) ** 2


def triangle(x: float):
    if x <= 1/2:
        return x
    else:
        return 1 - x


def inv_triangle(x: float):
    return 1 - triangle(x)


def constante(x: float, cst: float = 4.2):
    return cst


def impulse(x:float, start:float = .1, length:float = .3, val:float = 1):
    assert 0 < start < start + length < 1
    return val if start < x < start + length else 0


def abs_sin(X: float | ndarray | Iterable):
    return abs(nsin(X))


if __name__ == '__main__':
    graph_func(carre)

