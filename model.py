from utils import FullyConnectedLayer, OutputLayer, Dropout

class model:
    def __init__(self, input_dim, output_dim):
        self.layers = [
            FullyConnectedLayer(input_dim,256, activation='relu'),
            Dropout(0.1),
            FullyConnectedLayer(256,128, activation='relu'),
            OutputLayer(128,output_dim)
        ]  
        self.parameters = [
            param 
            for layer in self.layers if not isinstance(layer, Dropout) 
            for param in layer.get_param() 
        ]

    def train(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.is_train = True

    def eval(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.is_train = False
        
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
        
    def backward(self, labels):
        for i in range(len(self.layers) - 1, -1, -1):
            if i == (len(self.layers) - 1):
                dX = self.layers[i].backward(labels)    #Output层单独处理
            else:
                dX = self.layers[i].backward(dX)
