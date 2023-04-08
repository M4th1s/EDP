from model_EDP_ordre1 import ModelEDP
from math import sin

if __name__ == "__main__":
    model1 = ModelEDP(sin, lambda t, x: 1)
    #model1.graph_3d()