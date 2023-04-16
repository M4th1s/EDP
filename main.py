from model_EDP_ordre1 import ModelEDP, ModelEDP_a_cst
from numpy import sin

if __name__ == "__main__":
    #model1 = ModelEDP(sin, lambda t, x: 1)
    #model1.graph_3d()

    model2 = ModelEDP_a_cst(sin, 1)
    model2.graph_3d()