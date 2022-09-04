import numpy as np

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

    def compile(self, loss="mse"):
        '''
        Using the layers added to the sequential model, ties them together by matching the
        input sizes of each layer with the output sizes of the previous layer.

        Currently doesn't actually compile the model into a static graph for best execution performance,
        essentially acting like setting the `run_eagerly` flag to true in the TensorFlow compile.
        https://www.tensorflow.org/api_docs/python/tf/keras/Model
        '''
        prev_output_size = self.layers[0].output_size
        for i in range(1, len(self.layers)):
            self.layers[i].generate(prev_output_size)
            prev_output_size = self.layers[i].output_size

    # def fit(self):
# 

# Functions

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * np.mean(y_pred - y_true)