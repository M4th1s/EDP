from model_EDP_ordre1 import ModelEDP, ModelEDP_a_cst
from numpy import sin, pi

nsin = lambda X:  sin(X * 2 * pi)

if __name__ == "__main__":
    #model1 = ModelEDP(sin, lambda t, x: 1)
    #model1.graph_3d()

    model2 = ModelEDP_a_cst(nsin, 1)
    model2.graph_3d()
    #model2.graph_2d(nb_droites=8)
