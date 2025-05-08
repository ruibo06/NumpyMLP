import numpy as np

class param:
    def __init__(self, type, value : np.ndarray, grad : np.ndarray):
        self.type = type
        self.value = value
        self.grad = grad


#k:样本数       n:输入向量长度  m:输出向量长度
#X:输入向量     Z:输出向量      A:经过激活函数后的向量

# 定义激活函数      Z[k,m]
def relu(Z, backward = False): 
    mask = Z > 0
    if not backward:
        A = Z * mask
        return A
    else:
        return mask.astype(int)

def sigmoid(Z, backward = False):
    A = 1 / (np.exp(-Z) + 1)
    if not backward:
        return A
    else:
        dZ = A * (1 - A)
        return dZ

def tanh(Z, backward = False):
    A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
    if not backward:
        return A
    else:
        return 1 - A ** 2

def leakyrelu(Z, backward = False, alpha = 0.01):
    mask = Z > 0
    A = alpha * Z
    A[mask] = Z[mask]
    if not backward:
        return A
    else:
        dZ = 0.01 * np.ones(Z.shape)
        dZ[mask] = 1
        return dZ
    
activation_function_map = {
    "relu" : relu,
    "sigmoid" : sigmoid,
    "tanh" : tanh,
    "leakyrelu" : leakyrelu
}

#定义全连接层，     X[k,n]
class FullyConnectedLayer: 
    def __init__(self, input_dim, output_dim, activation='relu'):
        W = np.random.randn(input_dim, output_dim) * 0.01
        b = np.zeros((1, output_dim))
        self.W = param('w',W, np.zeros_like(W))
        self.b = param('b',b, np.zeros_like(b))
        self.activation = activation

    def forward(self, X):
        self.X = X
        self.Z = np.matmul(X, self.W.value) + self.b.value
        A = activation_function_map[self.activation](self.Z)
        return A
    
    def backward(self, dA):                                                         #dA[k,m]
        dZ = dA * activation_function_map[self.activation](self.Z, backward = True) #dZ[k,m]   
        self.W.grad= np.matmul(self.X.T, dZ) / self.X.shape[0]              #[n,k]*[k,m]   W_grad[n,m]
        self.b.grad = np.sum(dZ, axis=0, keepdims=True) / self.X.shape[0]   #b_grad[1,m]
        dX = np.matmul(dZ, self.W.value.T)                                  #[k,m]*[m,n]   dX[k,n]

        return dX
    
    def get_param(self):
        return self.W, self.b


#定义损失函数和Output层
def softmax(Z):                                     #Z[k,m]
    A = Z - np.max(Z, axis=1, keepdims=True)
    A = np.exp(A)
    sum = np.sum(A, axis=1, keepdims=True)          #sum[k,1]
    A = A / sum
    return A

def onehot(labels, label_num = 10):                 #labels[k,1]
    Y = np.zeros((labels.shape[0],label_num))       #Y[k,10]
    for i in range(labels.shape[0]):
        Y[i][labels[i][0]] = 1
    return Y

def CrossEntropyLoss(Y_predict, labels):
    Y = onehot(labels)                              #Y[k,10]
    ls = np.log(Y_predict + 1e-12) * Y  
    ls = -np.sum(ls, axis = 1)                      #ls[k,]
    ls = np.sum(ls) / Y.shape[0]
    return ls

class OutputLayer:
    def __init__(self, input_dim, output_dim):
        self.output_dim = output_dim
        W = np.random.randn(input_dim, output_dim) * 0.01
        b = np.zeros((1, output_dim))
        self.W = param('w',W, np.zeros_like(W))
        self.b = param('b',b, np.zeros_like(b))

    def forward(self, X):
        self.X = X
        Z = np.matmul(X, self.W.value) + self.b.value
        self.A = softmax(Z)
        return self.A
    
    def backward(self, labels):
        Y = onehot(labels, label_num=self.output_dim)
        dZ = self.A - Y                 #dZ[k,m]
        self.W.grad = np.matmul(self.X.T, dZ) / self.X.shape[0]
        self.b.grad = np.sum(dZ, axis=0, keepdims=True) / self.X.shape[0]

        dX = np.matmul(dZ, self.W.value.T)
        return dX
    
    def get_param(self):
        return self.W, self.b


#定义Dropout层
class Dropout:
    def __init__(self, dropout):
        assert 0 <= dropout <= 1
        self.dropout = dropout
        self.is_train = True        #is_train表示模型是否处于训练模式
    def forward(self, X):
        if not self.is_train:
            return X
        else:
            if self.dropout == 0:
                return X
            elif self.dropout == 1:
                return np.zeros_like(X)
            else:
                mask = np.random.rand(*X.shape) > self.dropout
                return X * mask / (1 - self.dropout)
            
    def backward(self, dX):
        return dX
    
    
#定义优化器
class Optimizer():
    def __init__(self, parameters, lr, weight_decay = 0.):
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        raise NotImplementedError
    
class SGDOptimizer(Optimizer):
    def step(self):
        for parameter in self.parameters:
            if  parameter.type == 'w':
                grad = parameter.grad + parameter.value * self.weight_decay
            else:
                grad = parameter.grad

            parameter.value -= self.lr * grad

class MomentumOptimizer(Optimizer):
    def __init__(self, parameters, lr, momentum = 0.9,  weight_decay = 0.):
        super().__init__(parameters, lr, weight_decay)
        self.momentum = momentum
        self.velocity = []
        self.__init_velocity()

    def __init_velocity(self):
        for parameter in self.parameters:
            self.velocity.append(np.zeros_like(parameter.grad))

    def step(self):
        for i in range(len(self.parameters)):
            if self.parameters[i].type == 'w':
                grad = self.parameters[i].grad + self.parameters[i].value * self.weight_decay
            else:
                grad = self.parameters[i].grad 
            self.velocity[i][:] = self.momentum * self.velocity[i] + grad

            self.parameters[i].value -= self.lr * self.velocity[i]

class AdamOptimizer(Optimizer):
    def __init__(self, parameters, lr, beta1 = 0.9, beta2 = 0.999, weight_decay=0, eps = 1e-6):
        super().__init__(parameters, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.velocity = []
        self.state = []
        self.beta1_t = 1
        self.beta2_t = 1
        self.__init_velocity_and_state()

    def __init_velocity_and_state(self):
        for parameter in self.parameters:
            self.velocity.append(np.zeros_like(parameter.grad))
            self.state.append(np.zeros_like(parameter.grad))

    def step(self):
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2                  #估计偏差修正

        for i in range(len(self.parameters)):
            if self.parameters[i].type == 'w':
                grad = self.parameters[i].grad + self.parameters[i].value * self.weight_decay
            else:
                grad = self.parameters[i].grad 
            self.velocity[i][:] = self.beta1 * self.velocity[i] + (1-self.beta1) * grad
            self.state[i][:] = self.beta2 * self.state[i] + (1-self.beta2) * grad ** 2

            v_cor = self.velocity[i] / (1 - self.beta1_t)
            s_cor = self.state[i] / (1 - self.beta2_t)

            self.parameters[i].value -= self.lr / (self.eps + np.sqrt(s_cor)) * v_cor


#定义DataLoader
class DataLoader:
    def __init__(self, X, y, batchsize = 64, shuffle = True):
        self.X = X
        self.y = y

        if shuffle:     #不能分开打乱!!!
            idx = np.arange(len(self.X))
            np.random.shuffle(idx)
            self.X = self.X[idx]
            self.y = self.y[idx]

        self.batchsize = batchsize
        self.start = 0
        self.end = X.shape[0]
        
    def __iter__(self):
        self.current = self.start
        return self
    
    def __next__(self):
        if self.current < self.end:
            start = self.current
            if self.current + self.batchsize - 1 < self.end:
                end = self.current + self.batchsize
            else:
                end = self.end
            self.current += self.batchsize
            return {'data':self.X[start:end][:], 'target':self.y[start:end][:]}
        else:
            raise StopIteration
        
    def __len__(self):
        return ((self.end - 1) // self.batchsize) + 1


#Xavier和He参数初始化
def xavier_uniform(weight : np.ndarray, gain = 1):
    range = gain * np.sqrt(6/(weight.shape[0] + weight.shape[1]))
    weight[:] = np.random.uniform(-range,range,weight.shape)

def xavier_normal(weight : np.ndarray, gain = 1):
    std = gain * np.sqrt(2/(weight.shape[0] + weight.shape[1]))
    weight[:] = np.random.normal(0, std, weight.shape)

def he_uniform(weight : np.ndarray, a = 0, mode = 'in'):
    assert mode == 'in' or mode == 'out'
    if mode == 'in':
        fan = weight.shape[0]
    else:
        fan = weight.shape[1]
    range = np.sqrt(6/((1+a**2) * fan))
    weight[:] = np.random.uniform(-range,range,weight.shape)

def he_normal(weight : np.ndarray, a = 0, mode = 'in'):
    assert mode == 'in' or mode == 'out'
    if mode == 'in':
        fan = weight.shape[0]
    else:
        fan = weight.shape[1]
    std = np.sqrt(2/((1+a**2) *fan))
    weight[:] = np.random.normal(0, std, weight.shape)