import numpy as np
import json

class DenseLayer: 

    """En caso de que se quiera usar este, solo lo toman, quitan el otro y ya"""
    def __init__(self,n_entries,n_neurons,activation):
        self.weights = np.array([[np.random.rand() *0.01 for _ in n_neurons] for _ in n_entries])
        self.biases = np.zeros(n_neurons)
        self.activation = activation

    def __init__(self,w,b,activation):
        self.weights = w
        self.biases = b
        self.activation = activation
    
    def sygmoid(x):
        return 1/(1+np.exp(-x))
    
    def softmax(self,x):
        exp_x = np.exp(x- np.max(x))
        return exp_x/ np.sum(exp_x,axis=0)
    
    def relu(self,x):
        return np.maximum(0,x)

    def leakyRelu(self,x, alpha = 0.01):
        return np.where(x>= 0, x, alpha*x)
    
    def step_function(self,x):
        return np.where(x>= 0,1,0)

    def telu(self,x, alpha=0.01):
        return np.where(x>= 0, x, alpha*(np.exp(x)-1))    

    def forward(self, en):
        if not isinstance(en, np.ndarray):
            raise ValueError("The var send is not an array")

        # Transfer function
        self.output = np.dot(en,self.weights) + self.biases

        # Choose of activation function 
        if self.activation == "sygmoid":
            self.output = self.sygmoid(self.output)
        elif self.activation == "tanh":
            self.output = self.tanh(self.output)
        elif self.activation == "softmax":
            self.output = self.softmax(self.output)
        elif self.activation == "relu":
            self.output = self.relu(self.output)

    def tanh(self,x):
	    return np.tanh(x)

class NeuralNet:
    """En caso de que se quiera usar este, solo lo toman, quitan el otro y ya"""
    def __init__(self,n_entry):
        self.layers = []
        self.entry = n_entry

    def __init__(self,file):
        self.layers = []
        if not isinstance(file,str):
            raise ValueError("The file sended need to be a direction string")
        
        with open(file, "r",encoding="utf-8") as f:
            datos = json.load(f)

        self.entry = len(datos["layers"][0]["W"])
        for i in datos["layers"]:
            self.layers.append(DenseLayer(w=np.array(i["W"]), b=np.array(i["b"]), activation=i["activation"]))


    def appendLayer(self,n_neurons,activation="relu"):
        if len(self.layers) == 0:
            self.layers.append(DenseLayer(n_entries=self.entry, n_neurons=n_neurons, activation=activation))
        else:
            prev_neurons = self.layers[-1].weights.shape[1]
            self.layers.append(DenseLayer(n_entries=prev_neurons, n_neurons=n_neurons, activation=activation))

    # array arg: False only show the argmax; True the complete array
    def solve_batch(self,input, array=False):
        if not isinstance(input, np.ndarray):
            raise ValueError("It will not function")
        
        if not len(input[0]) == self.entry:
            raise ValueError("It cant be used the input")
        
        self.layers[0].forward(input)
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output)

        if not array:
            return [np.argmax(l) for l in self.layers[-1].output]
        else :
            return self.layers[-1].output
    
    def solve(self,input, array=False):
        if not isinstance(input, np.ndarray):
            raise ValueError("It will not function")
        
        if not len(input) == self.entry:
            raise ValueError("It cant be used the input")
        
        self.layers[0].forward(input)
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output)

        if not array:
            return np.argmax(self.layers[-1].output) 
        else :
            return self.layers[-1].output
    
    def calc_precision(self,class_targets):
        pass
        


