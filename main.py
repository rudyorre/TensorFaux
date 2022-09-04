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

    nn.fit(X, Y, epochs=10000, learning_rate=0.01)
    Y_pred = nn.predict(X)

    for (y_true, y_pred) in zip(Y, Y_pred):
        print(y_true, y_pred)


if __name__ == '__main__':
    main()