
# PHASE 5: MLP from Scratch solving XOR problem

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(1)
W1 = np.random.randn(2,2)
b1 = np.zeros((1,2))
W2 = np.random.randn(2,1)
b2 = np.zeros((1,1))

lr = 0.1
epochs = 5000
losses = []
accuracies = []

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

for _ in range(epochs):
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    output = sigmoid(z2)

    loss = np.mean((y - output) ** 2)
    losses.append(loss)

    predictions = (output > 0.5).astype(int)
    accuracies.append(np.mean(predictions == y))

    d_output = (output - y) * sigmoid_derivative(output)
    dW2 = a1.T.dot(d_output)
    db2 = np.sum(d_output, axis=0)

    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(a1)
    dW1 = X.T.dot(d_hidden)
    db1 = np.sum(d_hidden, axis=0)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

plt.plot(losses)
plt.title("Loss Reduction")
plt.show()

plt.plot(accuracies)
plt.title("Accuracy Improvement")
plt.show()
