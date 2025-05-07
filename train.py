import numpy as np
from model import model
from utils import AdamOptimizer, DataLoader, CrossEntropyLoss, he_uniform
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy() / 255.0
y = mnist.target.to_numpy()
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = np.reshape(y_train,(-1,1))
y_test = np.reshape(y_test,(-1,1))

MyModel = model(784)
optimizer = AdamOptimizer(MyModel.parameters, lr=1e-3, weight_decay=1e-4)
loss_fn = CrossEntropyLoss
train_data = DataLoader(X_train, y_train, batchsize=64, shuffle=True)

for param in MyModel.parameters:
    if param.type == 'w':
        he_uniform(param.value)

epochs = 30
for epoch in range(epochs):
    print(f'epoch {epoch + 1}')

    MyModel.train()
    loss = 0
    accu = 0
    for batch in train_data:
        X = batch['data']
        y = batch['target']
        probs = MyModel.forward(X)
        MyModel.backward(y)
        loss += loss_fn(probs, y)
        preds = np.argmax(probs, axis=1, keepdims=True)
        accu += np.mean((preds == y), axis=0).item()
        optimizer.step()
    loss /= len(train_data)
    accu /= len(train_data)
    print(f"accu: {accu:.4f}\nloss: {loss:.4f}\n")

MyModel.eval()
probs = MyModel.forward(X_test)
preds = np.argmax(probs, axis=1, keepdims=True)
accu = np.mean((preds == y_test), axis=0).item()
print(f"Test accuracy: {accu:.4f}")
