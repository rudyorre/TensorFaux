import numpy as np

class Optimizer:
    def __init__(self):
        pass

class GD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def minimize(self, loss, loss_prime, X, Y, model):
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

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, batch_size=1):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

    def minimize(self, loss, loss_prime, X, Y, model):
        error = 0
        indices = np.random.choice(np.arange(0, len(X)), size=self.batch_size)
        for i in indices:
            x, y = X[i], Y[i]
            output = x

            # Forward prop
            output = model.predict([x])[0]

            # Error 
            error += loss(y, output)

            # Backward prop
            grad = loss_prime(y, output)
            for layer in reversed(model.layers):
                grad = layer.backward(grad, self.learning_rate)
        return error / self.batch_size
