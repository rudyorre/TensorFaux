from nn import Dense, Tanh, mse, mse_prime
import numpy as np

def main():
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    network = [
        Dense(2, 3),
        Tanh(),
        Dense(3, 1),
        Tanh()
    ]

    epochs = 10000
    learning_rate = 0.1

    for epoch in range(epochs):
        error = 0
        for (x, y) in zip(X, Y):
            output = x

            # Forward prop
            for layer in network:
                output = layer.forward(output)

            # Error 
            error += mse(y, output)

            # Backward prop
            grad = mse_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(X)
        print(f'{epoch + 1}/{epochs}, error={error}')

    for (x,y) in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        print(f'output: {output}, actual: {y}')



if __name__ == '__main__':
    main()