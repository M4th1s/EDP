from typing import Callable
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


        if abs(u0(0) - u0(1)) < 1e-9:
            self.u0 = u0
        else:
            raise ValueError("u0 doit être 1-périodique")
        self.a = a

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


class ModelEDP_a_cst(ModelEDP):

    def __init__(self,
                 u0: Callable,
                 a: float,
                 J: int = 10 ** 3,
                 delta_t: float = 10 ** -4,
                 N: int = 3 * 10 ** 3):

        super().__init__(u0, lambda t, x: a, J, delta_t, N)
        self.a = a


    def solve(self) -> (list[float], list[float], list[list[float]]):
        dx = 1 / self.J
        les_x = [(j + 1 / 2) * dx for j in range(self.J)]

        dt = self.delta_t
        les_t = [n * dt for n in range(self.N)]

        les_u_0_j = [self.u0(x) for x in les_x]
        les_u_n = [les_u_0_j]

        for _ in trange(1, self.N):
            u_prec = les_u_n[-1]  # u_n-1
            u_n_0 = u_prec[0] - self.a * dt / dx * (u_prec[0] - u_prec[-1])
            u_n = [u_n_0]

            for j in range(1, self.J):
                u_n_j = u_prec[j] - self.a * dt / dx * (u_prec[j] - u_prec[j-1])
                u_n.append(u_n_j)

            les_u_n.append(u_n)

        return les_t, les_x, les_u_n

    def solve_numpy(self):
        dx = 1 / self.J
        dt = self.delta_t

        self.X = np.array([(j + 1 / 2) * dx for j in range(self.J)])
        self.T = np.array([n * dt for n in range(self.N)])

        U = np.zeros((self.N, self.J), dtype="float128")

        for j, x in enumerate(self.X):
            U[0, j] = self.u0(x)

        for n in trange(1, self.N):
            for j in range(self.J):
                U[n, j] = U[n-1, j] - self.a * dt / dx * (U[n-1, j] - U[n-1, j-1])
            #print(max(U[n]))
            m = max(U[n])

        self.U = U
        return self.T, self.X, U

    def graph_3d(self):
        les_t, les_x, U = self.solve_numpy()

        X, T = np.meshgrid(les_x, les_t)

        #U_theo = self.u0(X - self.a * T) #TODO : make it work even for non numpy function

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(T, X, U)
        #ax.plot_surface(T, X, U_theo)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.show()

    def graph_2d(self, nb_droites: int = 4):
        """Graphe de u(t,.) pour t fixé"""
        les_t, les_x, U = self.solve_numpy()

        ind_quantile = int(self.N / (nb_droites + 1))
        for k in range(nb_droites):
            ind_k = k * ind_quantile
            t = les_t[ind_k]
            plt.plot(les_x, U[ind_k], label=f"u({t},*)")
        plt.show()

    def images_2d(self):
        les_t, les_x, U = self.solve_numpy()

        for n, t in enumerate(les_t):
            plt.clf()
            plt.plot(les_x, U[n])
            print(f"saving file {n=}")
            plt.savefig(f"./images/graphe_{n}.png")

