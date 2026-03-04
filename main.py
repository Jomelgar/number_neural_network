import sys
from neuralnetwork import NeuralNet
import numpy as np
import matplotlib.pyplot as plt

file = "./json/mnist_mlp_pretty.json"
if not isinstance(sys.argv[1],str):
    raise ValueError("The file must be a str for direction")

network = NeuralNet(file=file)
data = np.load(sys.argv[1])
images = data["images"].reshape(data["images"].shape[0],-1)/255

results = network.solve_batch(images)