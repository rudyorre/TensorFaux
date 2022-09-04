from nn import Input, Dense, Tanh, mse, mse_prime, Sequential
import numpy as np

def main():
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    network = [
        Input(2),
        Dense(3),
        Tanh(),
        Dense(1),
        Tanh()
    ]

    nn = Sequential()
    nn.add(Input(2))
    nn.add(Dense(3))
    nn.add(Tanh())
    nn.add(Dense(1))
    nn.add(Tanh())

    nn.compile()

    epochs = 10000
    learning_rate = 0.01

    for epoch in range(epochs):
        error = 0
        for (x, y) in zip(X, Y):
            output = x

            # Forward prop
            for layer in nn.layers:
                output = layer.forward(output)

            # Error 
            error += mse(y, output)

            # Backward prop
            grad = mse_prime(y, output)
            for layer in reversed(nn.layers):
                grad = layer.backward(grad, learning_rate)

        error /= len(X)
        if 100 * (epoch + 1) % epochs == 0:
            print(f'{epoch + 1}/{epochs}, error={error}')

    for (x,y) in zip(X, Y):
        output = x
        for layer in nn.layers:
            output = layer.forward(output)
        print(f'output: {output}, actual: {y}')



if __name__ == '__main__':
    main()