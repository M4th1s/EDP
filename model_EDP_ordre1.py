from typing import Callable
from tqdm import trange, tqdm
from time import sleep

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import os
import imageio

class ModelEDP():

    def __init__(self,
                 u0: Callable,
                 a: float = 1,
                 J: int = 10 ** 3,
                 #delta_t: float = 10 ** -5,
                 N: int = 10 ** 3,
                 cfl: float = .9, #Courant–Friedrichs–Lewy
                 ):


        if cfl > 0:
            self.delta_t = cfl / a / J
        else:
            raise ValueError("cfl doit être strictement positif")


        self.u0 = u0
        self.a = a

        self.N = N

        if J > 0:
            self.J = J
        else:
            raise ValueError("J doit être strictement positif")



    def solve(self):
        dx = 1 / self.J
        dt = self.delta_t

        self.X = np.array([(j + 1 / 2) * dx for j in range(self.J)])
        self.T = np.array([n * dt for n in range(self.N)])

        U = np.zeros((self.N, self.J))

        for j, x in enumerate(self.X):
            U[0, j] = self.u0(x)

        for n in trange(1, self.N, desc="Calcul des U_n_j"):
            for j in range(self.J):
                if (u_n_j := U[n-1, j] - self.a * dt / dx * (U[n-1, j] - U[n-1, j-1])) > 1e30:
                    raise OverflowError(f"explosion de la fonction en t={self.T[n]}")
                else:
                    U[n, j] = u_n_j

        self.U = U
        return self.T, self.X, U


    def results(self, regenerate:bool = False, save:bool = True):
        func_name = self.u0.__name__
        file_name = f"Data of {func_name} with param {self.a, self.J, self.delta_t, self.N}.npz"
        path = "./data/" + file_name

        if regenerate or not os.path.isfile(path):
            T, X, U = self.solve()
            if save:
                np.savez_compressed(path, T=T, X=X, U=U)
        else:
            with np.load(path) as data:
                T = data["T"]
                X = data["X"]
                U = data["U"]

        return T, X, U


    def graph_3d(self):
        les_t, les_x, U = self.results()

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
        les_t, les_x, U = self.results()

        ind_quantile = int(self.N / (nb_droites + 1))
        for k in range(nb_droites):
            ind_k = k * ind_quantile
            t = les_t[ind_k]
            plt.plot(les_x, U[ind_k], label=f"u({t},*)")
        plt.show()

    def images_2d(self):
        les_t, les_x, U = self.results()

        if not os.path.isdir("./images"):
            print("making dir")
            sleep(.1)
            os.makedirs("images")


        for n, t in tqdm(enumerate(les_t), total=len(les_t), desc="Sauvegarde des graphes"):
            plt.clf()
            plt.plot(les_x, U[n])
            plt.savefig(f"./images/graphe_{n}.png")

        images_to_gif(int(1/np.sqrt(self.delta_t)))

    def heat_map_2d(self):
        les_t, les_x, U = self.solve()

        plt.imshow(U, cmap='seismic')
        plt.colorbar()
        #plt.plot()
        plt.savefig("plot.png")
        print("Done")

def images_to_gif(fps:int, path:str = "images"):
    ls = sorted(os.listdir(path))
    images = []

    for file in tqdm(ls, "Chargement des images"):
        image = imageio.v2.imread("./" + path + "/" + file)
        images.append(image)


    imageio.mimsave("graph.gif", images)


if __name__ == "__main__":
    images_to_gif(50)
