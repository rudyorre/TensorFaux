import numpy as np
from tensorfaux.layers import Input, Dense, Tanh
from tensorfaux.models import Sequential
from tensorfaux.optimizers import SGD

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
        Dense(1),
        Tanh(),
    ])
    model.compile(optimizer=SGD(learning_rate=0.01, batch_size=3))

    # Training
    model.fit(X, Y, epochs=10000)

    # Prediction
    Y_pred = model.predict(X)
    for (y_true, y_pred) in zip(Y, Y_pred):
        print(f'Actual: {y_true}, Predicted: {y_pred}')


if __name__ == '__main__':
    main()
