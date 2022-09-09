import numpy as np

class Loss:
    def __init__(self):
        self.loss = lambda a,b: 0
        self.loss_prime = lambda a,b: 0
    
class MSE(Loss):
    def __init(self):
        def mse(y_true, y_pred):
            return np.mean(np.power(y_true - y_pred, 2))

        def mse_prime(y_true, y_pred):
            return 2 * np.mean(y_pred - y_true)

        self.loss = mse
        self.loss_prime = mse_prime

# Convolutional Neural Network (CNN) Functions:
# Functions taken from: https://www.youtube.com/watch?v=Lakz2MoHy6o&ab_channel=TheIndependentCode
class Binary_Cross_Entropy(Loss):
    def __init__(self):
        def binary_cross_entropy(y_true, y_pred):
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        def binary_cross_entropy_prime(y_true, y_pred):
            return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

        self.loss = binary_cross_entropy
        self.loss_prime = binary_cross_entropy_prime

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * np.mean(y_pred - y_true)

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)