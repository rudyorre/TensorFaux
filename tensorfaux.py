import numpy as np
import optimizers

class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_grad, alpha):
        pass

    def generate(self, input_size):
        self.input_size = input_size
        self.output_size = input_size

class Input(Layer):
    def __init__(self, input_size) -> None:
        self.input_size = input_size
        self.output_size = input_size

    def forward(self, input):
        return input

    def backward(self, output_grad, alpha):
        return output_grad

class Dense(Layer):
    def __init__(self, output_size) -> None:
        self.input_size = None
        self.output_size = output_size

    def generate(self, input_size):
        self.input_size = input_size
        self.weights = np.random.randn(self.output_size, self.input_size)
        self.bias = np.random.randn(self.output_size, 1)
    
    def forward(self, input):
        # if self.input_size == None:
        #     self.generate(input.shape[0])
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_grad, alpha):
        weights_grad = np.dot(output_grad, self.input.T)
        self.weights -= alpha * weights_grad
        self.bias -= alpha * output_grad
        return np.dot(self.weights.T, output_grad)

class Activation(Layer):
    def __init__(self, activation, activation_prime) -> None:
        self.activation = activation
        self.activation_prime = activation_prime
        self.input_size = None
        self.output_size = None
    
    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_grad, alpha):
        return np.multiply(output_grad, self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

class ReLU(Activation):
    '''
    ReLU prime: https://stackoverflow.com/questions/45021963/relu-prime-with-numpy-array
    '''
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0).astype(x.dtype)
        super().__init__(relu, relu_prime)

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)

class Linear(Activation):
    def __init__(self):
        linear = lambda x: 1.0 * x
        linear_prime = lambda x: 1.0
        super().__init__(linear, linear_prime)

class Softmax(Layer):
    def forward(self, input):
        '''
        Stable softmax upgrade: https://stackoverflow.com/questions/42599498/numerically-stable-softmax
        '''
        z = input - np.max(input)
        exps = np.exp(z)
        self.output = exps / np.sum(exps)
        return self.output
    
    def backward(self, output_grad, learning_rate):
        '''
        Softmax backprop derivation: https://www.youtube.com/watch?v=AbLvJVwySEo&ab_channel=TheIndependentCode
        '''
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_grad)

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
        loss = mse
        loss_prime = mse_prime
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

# Functions

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * np.mean(y_pred - y_true)