A simple neural network implementation which utilizes a simimilar API to TensorFlow's Sequential models.

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

* NumPy

```
pip3 install -r requirements.txt
```

### Sample Usage

Below is an example of a neural network with two `Dense()` layers using `Tanh()` activation functions learning the XOR function. Although seemingly trivial, the XOR function isn't [linearly separable](https://medium.com/@lucaspereira0612/solving-xor-with-a-single-perceptron-34539f395182#:~:text=Geometrically%2C%20this%20means%20the%20perceptron,single%20hyperplane%20to%20separate%20it.), meaning linear models such as [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) and single-layer [perceptrons](https://en.wikipedia.org/wiki/Perceptron) cannot learn XOR.

```python
from nn import Sequential, Input, Dense, Tanh

np.random.seed(42)

# XOR input/output data
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Model instantiation
model = Sequential([
    Input(2),
    Dense(3),
    Tanh(),
    Dense(1),
    Tanh(),
])
model.compile()

# Model training
model.fit(X, Y, epochs=10000, learning_rate=0.01)

# Predict
Y_pred = nn.predict(X)
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

# API
## `Sequential`
```python
nn.Sequential(
    layers=[]
)
```

Function|Description
-|-
`__init__`|Instantiates a new `Sequential` model. If given a list of layers, it will add these to the model, similar to `add()`.
`add`|Add a single layer to the model.
`compile`|Takes the added layers and the parameters of `compile` to generate a trainable model.
`fit`|After compilation, `fit()` trains the model on its inputs and outputs.
`predict`|Makes predictions after fitting to the data. Takes in a subset of the input data to make a prediction.

## Layers
### `Layer` Layer
Abstract class for the layers API. This shouldn't be used in an instance of a model.

### `Input` Layer
This should always be the first layer of the `Sequential` model. Since the other layers take in an explicit `output_size` as their input, they infer their `input_size` from the previous layer's `output_size`. This means we must declare the model's first `input_size`.

### `Dense` Layer
Just your regular densely-connected NN layer. At the moment, the dense layer only performs the dot product and bias addition, but no activation function. The activation function is out-sourced to the `Activation` layers.

### `Activation` Layer
Applies an activation function to an output.

### `Tanh` Layer
Hyperbolic tangent activation function.

## Acknowledgements
- [TensorFlow Brand Guidelines](https://www.tensorflow.org/extras/tensorflow_brand_guidelines.pdf)
- [@allejo/jekyll-docs-theme](https://github.com/allejo/jekyll-docs-theme)
- [Packaging Projects Tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
