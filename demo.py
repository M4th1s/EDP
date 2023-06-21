import numpy as np


def solve(u0,
          a: float = 1,
          J: int = 200,
          delta_t: float = .0001,
          N: int = 1000,
          ):
    
    dx = 1 / J
    dt = delta_t

    X = np.array([(j + 1 / 2) * dx for j in range(J)])
    T = np.array([n * dt for n in range(N)])

    U = np.zeros((N, J))

    for j, x in enumerate(X):
        U[0, j] = u0(x)

    for n in range(1, N, desc="Calcul des U_n_j"):
        for j in range(J):
            if (u_n_j := U[n-1, j] - a * dt / dx * (U[n-1, j] - U[n-1, j-1])) > 1e30:
                raise OverflowError(f"explosion de la fonction en t={T[n]}")
            else:
                U[n, j] = u_n_j

    return T, X, U

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def graph_3d(u0, a, J, delta_t, N):
    les_t, les_x, U = solve(u0, a, J, delta_t, N)

    X, T = np.meshgrid(les_x, les_t)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(T, X, U)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.show()


def graph_2d(u0, a, J, delta_t, N, nb_droites: int = 4):
    """Graphe de u(t,.) pour t fixé"""
    les_t, les_x, U = solve(u0, a, J, delta_t, N)

    ind_quantile = int(N / (nb_droites + 1))

    for k in range(nb_droites):
        ind_k = k * ind_quantile
        t = les_t[ind_k]
        plt.plot(les_x, U[ind_k], label=f"u({t},*)")
    plt.show()


def heat_map_2d(u0, a, J, delta_t, N):
    les_t, les_x, U = solve(u0, a, J, delta_t, N)

    plt.imshow(U, cmap='seismic')
    plt.colorbar()
    plt.plot()

