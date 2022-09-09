import numpy as np
from scipy import signal

class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_grad, alpha):
        pass

class Dense(Layer):
    def __init__(self, input_shape, output_shape) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.randn(self.output_shape, self.input_shape)
        self.bias = np.random.randn(self.output_shape, 1)
    
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
    def __init__(self, stable=False):
        self.stable=stable

    def forward(self, input):
        '''
        Stable softmax upgrade: https://stackoverflow.com/questions/42599498/numerically-stable-softmax
        '''
        z = input
        if self.stable:
            z -= np.max(z)
        exps = np.exp(z)
        self.output = exps / np.sum(exps)
        return self.output
    
    def backward(self, output_grad, learning_rate):
        '''
        Softmax backprop derivation: https://www.youtube.com/watch?v=AbLvJVwySEo&ab_channel=TheIndependentCode
        '''
        n = np.size(self.output)
        # print(f'n: {n}, self.output.T: {self.output.T.shape}, self.output: {self.output.shape} output_grad: {output_grad.shape}')
        return np.dot((np.identity(n) - self.output.T) * self.output, output_grad)

# Convolutional Neural Network Layers:
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_grad, learning_rate):
        kernels_grad = np.zeros(self.kernels_shape)
        input_grad = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_grad[i, j] = signal.correlate2d(self.input[j], output_grad[i], "valid")
                input_grad[j] += signal.convolve2d(output_grad[i], self.kernels[i, j], "full")
        
        self.kernels -= learning_rate * kernels_grad
        self.biases -= learning_rate * output_grad
        return input_grad

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_grad, learning_rate):
        return np.reshape(output_grad, self.input_shape)