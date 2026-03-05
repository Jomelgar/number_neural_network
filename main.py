import sys
from neuralnetwork import NeuralNet
import numpy as np
import json
import interactive_menu as m

# Esto Inge, solo es para cargar el json
FILE = "./json/mnist_mlp_pretty.json"


if not isinstance(sys.argv[1],str):
    raise ValueError("The file must be a str for direction")

network = NeuralNet()
network.load(FILE)

data = np.load(sys.argv[1])
images = data["images"].reshape(data["images"].shape[0],-1)/255
labels = data["labels"]

results = network.solve_batch(images)
dictionary = network.calc_precision(labels, results)
menu = m.Menu(dictionary,images,results,labels)
menu.show()



