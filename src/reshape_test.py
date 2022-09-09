from tensorfaux.layers import Reshape
import numpy as np

def main():
    data = np.arange(12)
    layer = Reshape(input_shape=(9), output_shape=(3,3))
    print(layer.forward(data))

if __name__ == '__main__':
    main()