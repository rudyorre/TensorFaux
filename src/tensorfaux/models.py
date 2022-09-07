import numpy as np
from tensorfaux import optimizers
from tensorfaux import util

class Sequential:
    def __init__(self, layers=[]):
        '''
        Sequential models should always start with an `Input` layer.
        '''
        self.layers = []
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss="mse", optimizer=optimizers.GD()):
        '''
        Using the layers added to the sequential model, ties them together by matching the
        input sizes of each layer with the output sizes of the previous layer.

        Currently doesn't actually compile the model into a static graph for best execution performance,
        essentially acting like setting the `run_eagerly` flag to true in the TensorFlow compile.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model
        '''
        self.optimizer = optimizer
        prev_output_size = self.layers[0].output_size
        for i in range(1, len(self.layers)):
            self.layers[i].generate(prev_output_size)
            prev_output_size = self.layers[i].output_size

    def fit(self, X, Y, epochs=1, verbose=False, verbose_count=10):
        loss = util.mse
        loss_prime = util.mse_prime
        for epoch in range(epochs):
            error = self.optimizer.minimize(loss, loss_prime, X, Y, self)
            if verbose and verbose_count * (epoch + 1) % epochs == 0:
                print(f'{epoch + 1}/{epochs}, error={error}')

    def predict(self, X):
        y_pred = []
        for x in X:
            output = x
            for layer in self.layers:
                output = layer.forward(output)
            y_pred.append(output)
        return np.array(y_pred)