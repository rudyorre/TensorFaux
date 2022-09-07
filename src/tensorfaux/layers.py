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