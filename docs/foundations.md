# Mathematical Foundations

Brief discussions on the mathematical components of Conv-TasNet are shown below.

## ReLU Activation

TensorFlow [docs](https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu) <br>
Keras [source code](https://github.com/keras-team/keras/tree/v2.7.0/keras/activations.py#L273-L311)

Rectified Linear Unit Activation layer sets all values of $x$ which are less than 0 to 0, and all values of $x$ which are greater than or equal to 0 remain as $x$ or, in the case of vectors $x_i$.

$$
f(x)= \left\{
  \begin{array}{lr} 
      x, & x \geq 0 \\
      0, & x < 0 
      \end{array}
\right.
$$


## Sigmoid Activation

TensorFlow [docs](https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid) <br>
Keras [source code](https://github.com/keras-team/keras/tree/v2.7.0/keras/activations.py#L376-L406)

The sigmoid activation layers bounds inputs within the range [-1, 1].

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

## PReLU

TensorFlow [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/PReLU) <br>
Keras [source code](https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/advanced_activations.py#L90-L180)

Parametric Rectified Linear Unit Activation layer sets all values of $x$ which are less than 0 to $\alpha x$, and all values of $x$ which are greater than or equal to 0 remain as $x$ or, in the case of vectors $x_i$.

$$
f(x)= \left\{
  \begin{array}{lr} 
      x, & x \geq 0 \\
      \alpha x, & x < 0 
      \end{array}
\right.
$$


## Layer Normalization

TensorFlow [docs](https://github.com/keras-team/keras/blob/2c48a3b38b6b6139be2da501982fd2f61d7d48fe/keras/layers/convolutional.py#L390-L527) <br>
Keras [source code](https://github.com/keras-team/keras/blob/2c48a3b38b6b6139be2da501982fd2f61d7d48fe/keras/layers/normalization/layer_normalization.py#L29-L363)

## 1D Convolution

TensorFlow [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D) <br>
Keras [source code](https://github.com/keras-team/keras/blob/2c48a3b38b6b6139be2da501982fd2f61d7d48fe/keras/layers/convolutional.py#L390-L527)


## 1D Convolution Transpose

TensorFlow [docs](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1DTranspose) <br>
Keras [source code](https://github.com/keras-team/keras/tree/v2.7.0/keras/layers/convolutional.py#L843-L1088)