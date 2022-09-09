import numpy as np
from tensorfaux.layers import *
from tensorfaux.models import Sequential
from tensorfaux.optimizers import GD, SGD
from tensorfaux.util import MSE, Binary_Cross_Entropy
import matplotlib.pyplot as plt

def main():
    np.random.seed(42)

    # Data
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    # Instantiation
    model = Sequential([
        Input(2),
        Dense(3),
        Tanh(),
        Dense(3),
        Tanh(),
        Dense(1),
        Tanh(),
    ])
    model.compile(loss=MSE(), optimizer=GD(learning_rate=0.01))
    running_error1 = model.fit(X, Y, epochs=10000, verbose=True, verbose_count=100)
    model.compile(optimizer=SGD(learning_rate=0.01, batch_size=10))
    running_error2 = model.fit(X, Y, epochs=10000, verbose=True, verbose_count=100)
    fig, ax = plt.subplots()
    ax.plot(np.arange(running_error1.shape[0]), running_error1)
    ax.plot(np.arange(running_error2.shape[0]), running_error2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Iteration Number versus Loss")
    plt.show()
    
    # GD().minimize(mse, mse_prime, X, Y, model)
    # error = 0
    # output = model.predict([X[0]])[0]
    # error = mse(Y[0], output)
    # grad = mse_prime(Y[0], output)
    # print(grad)
    Y_pred = model.predict(X)
    for (y_true, y_pred) in zip(Y, Y_pred):
        print(f'Actual: {y_true}, Predicted: {y_pred}')
'''
    error = 0
    for (x, y) in zip(X, Y):
        output = x

        # Forward prop
        output = model.predict([x])[0]

        # Error 
        error += loss(y, output)

        # Backward prop
        grad = loss_prime(y, output)
        for layer in reversed(model.layers):
            grad = layer.backward(grad, self.learning_rate)
    return error / len(X)
'''

def main1():
    np.random.seed(42)

    # Data
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    # Instantiation
    model = Sequential([
        Input(2),
        Dense(3),
        Sigmoid(),
        Dense(2),
        # Softmax(stable=False),
        ReLU(),
    ])
    model.compile(optimizer=GD(learning_rate=0.01))

    # Training
    model.fit(X, Y, epochs=10000, verbose=True, verbose_count=100)
    # np.dot((np.identity(n) - self.output.T) * self.output, output_grad)

    # Prediction
    Y_pred = model.predict(X)
    for (y_true, y_pred) in zip(Y, Y_pred):
        print(f'Actual: {y_true}, Predicted: {y_pred}')


if __name__ == '__main__':
    main()
