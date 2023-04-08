from typing import Callable
from inspect import signature
from tqdm import trange

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class ModelEDP:

    def __init__(self,
                 u0: Callable,
                 a: Callable,
                 J: int = 10 ** 3,
                 delta_t: float = 10 ** -3,
                 N: int = 10 ** 3,
                 ):
        if delta_t > 0:
            self.delta_t = delta_t
        else:
            raise ValueError("delta_t doit être strictement positif")

        try:
            sig_u0 = signature(u0)
            if len(sig_u0.parameters) == 1:
                self.u0 = u0
            else:
                raise TypeError
        except TypeError:
            raise TypeError("u0 est une fonction d'une variable (x)")

        try:
            sig_a = signature(a)
            if len(sig_a.parameters) == 2:
                self.a = a
            else:
                raise TypeError
        except TypeError:
            raise TypeError("a est une fonction de deux variables (t, x)")


        self.N = N

        if J > 0:
            self.J = J
        else:
            raise ValueError("J doit être strictement positif")

    def solve(self) -> (list[float], list[float], list[list[float]]):
        dx = 1 / self.J
        les_x = [(j + 1 / 2) * dx for j in range(self.J)]

        dt = self.delta_t
        les_t = [n * dt for n in range(self.N)]

        les_u_0_j = [self.u0(x) for x in les_x]
        les_u_n = [les_u_0_j]

        for n in trange(1, self.N):
            u_prec = les_u_n[-1]  # u_n-1
            u_n_0 = u_prec[0] - self.a(les_t[n], les_x[0]) * dt / dx * (u_prec[0] - u_prec[-1])
            u_n = [u_n_0]

            for j in range(1, self.J):
                u_n_j = u_prec[j] - self.a(les_t[n], les_x[j]) * dt / dx * (u_prec[j] - u_prec[j-1])
                u_n.append(u_n_j)

            les_u_n.append(u_n)

        return les_t, les_x, les_u_n

    def u(self, t, x, les_t, les_x, les_u_n):
        n = les_t.index(t)
        j = les_x.index(x)
        return les_u_n[n][j]

    def graph_3d(self):
        les_t, les_x, les_u_n = self.solve()

        T, X = np.meshgrid(les_t, les_x)
        U = np.matrix(les_u_n)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(T, X, U)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.show()
