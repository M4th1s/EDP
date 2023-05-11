from model_EDP_ordre1 import ModelEDP
import functions as f





if __name__ == "__main__":
    #model1 = ModelEDP(sin, lambda t, x: 1)
    #model1.graph_3d()

    model2 = ModelEDP(f.nsin, 1)
    #model2.graph_3d()
    #model2.graph_2d(nb_droites=7)
    #model2.images_2d()
    model2.heat_map_2d()

