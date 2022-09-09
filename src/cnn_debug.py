import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from tensorfaux.layers import Dense, Convolutional, Sigmoid, Reshape
from tensorfaux.models import Sequential
from tensorfaux.optimizers import GD

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

def main():
    # load MNIST from server, limit to 100 images per class since we're not training on GPU
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 100)
    x_test, y_test = preprocess_data(x_test, y_test, 100)

    # CNN
    model = Sequential([
        Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=5),
        Sigmoid(),
        Reshape(input_shape=(5, 26, 26), output_shape=(5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid()
    ])
    model.compile(loss='binary_cross_entropy', optimizer=GD(learning_rate=0.1))

    # Train
    model.fit(x_train, y_train, epochs=20, verbose=True, verbose_count=100)

    # Test
    for x, y in zip(x_test, y_test):
        output = model.predict([x])
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")

if __name__ == '__main__':
    main()