---
layout: full
homepage: true
disable_anchors: true
description: <h4>/ˈtensərfō/</h4><br /><h5>A knock-off of TensorFlow's more basic deep learning features.</h5>
---

A neural network implementation which utilizes a simimilar API to [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras), the high-level API of [TensorFlow](https://www.tensorflow.org/). Everything here is written from scratch in [Python](https://www.python.org/) and [Numpy](https://numpy.org/), which allows these complex models to be more interpretable.

<!---
> I've never seen a more mediocre theme it actually hurts my insides.
>
> ~ _Anonymous_, 2022


This theme is designed for writing documentation websites instead of having large unmaintainable README files or several markdown files inside of a folder in a repository.
-->

<div class="row">
<div class="col-lg-6" markdown="1">

## Installation
{:.mt-lg-0}

This library is designed to be as simple as possible, so the only Python dependency is Numpy. Simply running the code should be enough, however, initializing a [virtual environment](https://docs.python.org/3/library/venv.html) in the repo directory would be best practice.

```zsh
# Install repository
git clone https://github.com/rudyorre/TensorFaux.git

# Install Dependencies
cd TensorFaux
pip3 install -r requirements.txt
```

### Why did I make this?

The purpose of this project was for me to better understand what happens behind the scenes when I'm working with a high-level deep learning package such as [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras). The goal is to mimic most of the fundamental features of Keras.

</div>
<div class="col-lg-6" markdown="1">

## Features
{:.mt-lg-0}

Despite being a small library, the simplicity and flexibility of this library makes it a (potentially) good learning tool for better understanding the fundamentals of deep learning.

### Simple

Not only does the API reduce cognitive load in its simplicity, the respective code is also built to be as simple as possible. Instead of having to dive into the low-level tensor operations, exportable graphs and hardware support, you can look directly at the fundamental linear algebra and calculus that brings deep learning to life.

### Flexibility

Due to its simplicity, the code could be easily modified to support new features and bug fixes. This flexibility allows anyone to enhance this project with a relatively gentle learning curve.

</div>
</div>

## Sample Usage

Below is an example of a neural network with two `Dense()` layers using `Tanh()` activation functions, optimized with [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), learning the XOR function. Although seemingly trivial, the XOR function isn't [linearly separable](https://medium.com/@lucaspereira0612/solving-xor-with-a-single-perceptron-34539f395182#:~:text=Geometrically%2C%20this%20means%20the%20perceptron,single%20hyperplane%20to%20separate%20it.), meaning linear models such as [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) and single-layer [perceptrons](https://en.wikipedia.org/wiki/Perceptron) cannot learn XOR.

```python
import numpy as np
from tensorfaux.layers import Input, Dense, Tanh
from tensorfaux.models import Sequential
from tensorfaux.optimizers import SGD

np.random.seed(42)

# Data
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Instantiation
model = Sequential([
    Input(2),
    Dense(3),
    Tanh(),
    Dense(1),
    Tanh(),
])
model.compile(optimizer=SGD(learning_rate=0.01, batch_size=3))

# Training
model.fit(X, Y, epochs=10000)

# Prediction
Y_pred = model.predict(X)
for (y_true, y_pred) in zip(Y, Y_pred):
    print(f'Actual: {y_true}, Predicted: {y_pred}')
```
Output:
```
Actual: [[0]], Predicted: [[0.0003956]]
Actual: [[1]], Predicted: [[0.97018558]]
Actual: [[1]], Predicted: [[0.97092169]]
Actual: [[0]], Predicted: [[0.00186825]]
```
