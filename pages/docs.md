---
layout: page
title: Docs
permalink: /docs/
order: 1
---

Detailed documentation and user guides for TensorFaux. In all honesty, since the API is so similar to [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras), most of the documentation here is copied straight from their docs.

## Models

### Sequential

The sequential model ties all of the layers together under one simple API. While you could hard-code forward and backward propagation using a list of layers (`list(Layer)`), the `Sequential` API handles compilation, training and prediction.

`Sequential(layers=[])`|The constructor takes an optional argument of a list of layers.
`add(layer)`|Adds additional layers on top of the initial list of layers.
`compile(loss, optimizer)`|After adding the necessary layers, defines the loss function and optimizer that will be used for training.
`fit(X, Y, epochs=1, verbose=False, verbose_count=10)`|Takes in the input data and labels for training, along with a few optional arguments: `epochs` is the number of times to perform a forward and backward propagation, `verbose` controls whether or not the running loss value gets outputted to console, and `verbose_count` controls how many loss values get outputted to the console.


## Layers
While the `Sequential` model holds the API used for training the neural network, the layers are

### Layer
This is the class from which all layers inherit. A layer is a callable object that takes a numpy array and outputs a numpy array. Despite the functional nature of layers, they can also contain a state that can change over time, typically in the form of `weights`. All layers share two main functions: `forward()` and `backward()`.

`forward(input)`|Propagating forward to move data from the input layer to the output layer. These involve a variety of mathematical operations that eventually lead to a final output, also known as the prediction.
`backward(output_grad, learning_rate)`|When propagating data backwards, instead of the original data being used directly, an output gradient is computed from the output, which is propagated backwards through layers in a series of gradient calculations.

### Dense
Just your regular densely-connected NN layer. Initializes with random weights. `Dense` implements the operation: `output = dot(input, kernel) + bias` where kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer.

### Convolutional
2D convolution layer (e.g. spatial convolution over images). This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.

### Reshape
Layer that reshapes inputs into the given shape.

**Example:**
```python
>>> from tensorfaux.layers import Reshape
>>> import numpy as np
>>> data = np.arange(12)
>>> layer = Reshape(input_shape=(12), output_shape=(3,4))
>>> layer.forward(data)
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> layer = Reshape(input_shape=(12), output_shape=(6,2))
>>> layer.forward(data)
array([[ 0,  1],
       [ 2,  3],
       [ 4,  5],
       [ 6,  7],
       [ 8,  9],
       [10, 11]])
```

### Activation
Applies an activation function to an output. This is the class from which most activation layers inherit.

### Hyperbolic Tangent (Tanh)
Hyperbolic tangent activation function.

### Rectified Linear Unit (ReLU)
Applies the rectified linear unit activation function. With default values, this returns the standard ReLU activation: max(x, 0), the element-wise maximum of 0 and the input tensor.

### Sigmoid
Sigmoid activation function, `sigmoid(x) = 1 / (1 + exp(-x))`. Applies the sigmoid activation function. For small values (<-5), sigmoid returns a value close to zero, and for large values (>5) the result of the function gets close to 1.

Sigmoid is equivalent to a 2-element Softmax, where the second element is assumed to be zero. The sigmoid function always returns a value between 0 and 1.

### Linear
Linear activation function (pass-through).

### Softmax
Softmax converts a vector of values to a probability distribution. The elements of the output vector are in range (0, 1) and sum to 1. Each vector is handled independently.

## Optimizers

### Gradient Descent (GD)
Classic Gradient descent. The most basic iterative algorithm for finding a local minima.

### Stochastic Gradient Descent (SGD)
An iterative method which acts as a superclass to `GD`, instead of being constrained to using the full dataset for each pass-through, we can use `batch_size` (ranging from 1 to the number of inputs) to determine how many points of data are used in each pass. With each `epoch`, this optimizer will randomly choose inputs to use for forward and backward propagation.

Another argument, `momentum`, will be added eventually.

## Utility

### Mean Squared Error (mse)
Computes the mean squared error between labels and predictions. In statistics, the mean squared error (MSE) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.

### Binary Cross Entropy (binary_cross_entropy)
Computes the cross-entropy loss between true labels and predicted labels. Binary cross entropy is the negative average of the log of corrected predicted probabilities used for classification problems ([Wikipedia](https://en.wikipedia.org/wiki/Cross_entropy)).