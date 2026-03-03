from neuralnetwork import NeuralNet
import numpy as np
import matplotlib.pyplot as plt

file = "./json/mnist_mlp_pretty.json"
data_url = "./assets/mnist_custom_ds.npz"

network = NeuralNet(file=file)
data = np.load(data_url)
images = data["images"].reshape(10,-1)/255

results = network.solve_batch(images)
print(results)

plt.figure(figsize=(10, 4))

for i in range(len(data["images"])):
    plt.subplot(2, 5, i+1)
    plt.imshow(data["images"][i], cmap="gray")
    plt.title(f"Pred: {results[i]}")
    plt.axis("off")
plt.show()