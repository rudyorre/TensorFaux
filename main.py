from nn import Input, Dense, Softmax, Tanh, mse, mse_prime, Sequential, Softmax
import numpy as np
import nn

def main():
    np.random.seed(42)

    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
    print(X)
    network = [
        Input(2),
        Dense(3),
        Tanh(),
        Dense(1),
        Tanh()
    ]

    model = Sequential()
    model.add(Input(2))
    model.add(Dense(3))
    model.add(Tanh())
    model.add(Dense(1))
    # nn.add(Softmax())
    model.add(Tanh())

    model.compile(optimizer=nn.optimizers.GD(learning_rate=0.01))

    model.fit(X, Y, epochs=10000, verbose=True)
    Y_pred = model.predict(X)

    for (y_true, y_pred) in zip(Y, Y_pred):
        print(f'Actual: {y_true}, Predicted: {y_pred}')


if __name__ == '__main__':
    main()