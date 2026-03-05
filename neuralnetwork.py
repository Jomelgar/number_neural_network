import numpy as np
import json

class DenseLayer: 

    def __init__(self,n_entries=0,n_neurons=0,activation="sygmoid"):
        self.weights = np.array([[np.random.rand() *0.01 for _ in range(n_neurons)] for _ in range(n_entries)])
        self.biases = np.zeros(n_neurons)
        self.activation = activation

    def load(self,w,b,activation):
        self.weights = w
        self.biases = b
        self.activation = activation
        return self
    
    def sygmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def softmax(self,x):
        exp_x = np.exp(x- np.max(x))
        return exp_x/ np.sum(exp_x,axis=0)

    def softmax_batch(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
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

        self.output = np.dot(en,self.weights) + self.biases

        if self.activation == "sygmoid":
            self.output = self.sygmoid(self.output)
        elif self.activation == "tanh":
            self.output = self.tanh(self.output)
        elif self.activation == "softmax":
            if isinstance(self.output[0],np.ndarray):
                self.output = self.softmax_batch(self.output)
            else: 
                self.output = self.softmax(self.output)
        elif self.activation == "relu":
            self.output = self.relu(self.output)

    def tanh(self,x):
        return np.tanh(x)

class NeuralNet:
    
    def __init__(self,n_entry=1):
        self.layers = []
        self.entry = n_entry

    def load(self,file):
        self.layers = []
        if not isinstance(file,str):
            raise ValueError("The file sended need to be a direction string")
        
        with open(file, "r",encoding="utf-8") as f:
            datos = json.load(f)

        self.entry = len(datos["layers"][0]["W"])
        for i in datos["layers"]:
            var = DenseLayer()
            self.layers.append(var.load(w=np.array(i["W"]), b=np.array(i["b"]), activation=i["activation"]))


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
    
    def calc_precision(self, class_targets, predict_targets):
        data = {"len": 0,"precision": 1, "fails": []}
        data.update({f"{i}": {"len": 0,"fails": [], "precision": 1} for i in range(10)})

        for i, j in enumerate(class_targets):
            predicted = predict_targets[i]
            data["len"] += 1
            data[f"{j}"]["len"] += 1
            if j != predicted:
                data[f"{j}"]["fails"].append(i)
                data["fails"].append(i)

        data["precision"] = 1 - len(data["fails"]) / data["len"]
        for i in range(10):
            data[f"{i}"]["precision"] = 1- len(data[f"{i}"]["fails"])/data[f"{i}"]["len"]
        return data


