import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils

from tensorfaux.layers import Dense, Convolutional, Sigmoid, Reshape
from tensorfaux.models import Sequential
from tensorfaux.optimizers import GD, SGD

def preprocess_data(x, y, limit):
    # zero_index = np.where(y == 0)[0][:limit]
    # one_index = np.where(y == 1)[0][:limit]
    # all_indices = np.hstack((zero_index, one_index))
    # all_indices = np.random.permutation(all_indices)
    # x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

def main():
    # load MNIST from server, limit to 100 images per class since we're not training on GPU
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train = preprocess_data(x_train, y_train, 100)
    x_test, y_test = preprocess_data(x_test, y_test, 100)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # CNN
    model = Sequential([
        Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=5),
        Sigmoid(),
        Reshape(input_shape=(5, 26, 26), output_shape=(5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 10),
        Sigmoid()
    ])
    model.compile(loss='binary_cross_entropy', optimizer=SGD(learning_rate=0.01, batch_size=10))

    # Train
    running_error = model.fit(x_train, y_train, epochs=10000, verbose=True, verbose_count=100)

    fig, ax = plt.subplots()
    ax.plot(np.arange(running_error.shape[0]), running_error)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Iteration Number versus Loss")
    plt.show()

    # Test
    correct = 0
    for x, y in zip(x_test, y_test):
        output = model.predict([x])
        if np.argmax(output) == np.argmax(y):
            correct += 1
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    print(f'Accuracy: {correct / len(x_test)}')

if __name__ == '__main__':
    main()