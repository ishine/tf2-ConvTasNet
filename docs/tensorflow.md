# Customizing TensorFlow

[TensorFlow](https://www.tensorflow.org) is a robust machine learning platform for end-to-end projects. [Keras](https://keras.io) is a deep learning API that `enables fast experimentation` and uses TensorFlow under the hood. Keras is the recommended way to interact with TensorFlow Modules, Tensors, and Variables while still enabling engineers to customize and extend TensorFlow Core and other libraries such as TensorFlow Agents.

It is important to understand TensorFlow as the ecosystem and Keras as a tool within that ecosystem. Although one can use lower level TensorFlow classes such as Tensor and Module, using the available Keras Model, Sequential, and Layer classes will cut down on development time of certain utility methods available to these classes which are not native to Tensor and Module. 

Engineers can leverage additional TensorFlow tools and modules such as those available in [tf.math](https://www.tensorflow.org/api_docs/python/tf/math), [tf.linalg](https://www.tensorflow.org/api_docs/python/tf/linalg), [tf.signal](https://www.tensorflow.org/api_docs/python/tf/signal), [tf.nn](https://www.tensorflow.org/api_docs/python/tf/nn), and [TensorFlow Probability](https://www.tensorflow.org/probability) to create a NumPy-like development process while reducing dependency management issues.

> TensorFlow does make use of NumPy. It is recommended to allow the TensorFlow install to determine the NumPy version rather than actively managing the NumPy install independently. If using Pandas or xarray in the same conda env or virtualenv, one should pin the Pandas version which does not cause NumPy versioning conflicts.

Custom Model and Layer classes are implemented with basic inheritance principles:

```py
from tensorflow import keras


class CustomModel(keras.Model):
    """A custom Keras Model class
    """
    def __init__(self):
        super(CustomModel, self).__init__()

    def call(self):
        """Forward pass
        """
        pass


class CustomLayer(keras.layers.Layer):
    """A custom Keras Layer class
    """
    def __init__(self):
        super(CustomLayer, self).__init__()

    def build(self):
        """Create weights
        """
        pass

    def call(self):
        """Forward pass
        """
        pass
```

## Keras Model

Keras Model overview [docs](https://keras.io/api/models/model/#model-class) <br>
Keras Model class [docs](https://keras.io/api/models/model/#model-class)

## Keras Sequential

Keras [docs](https://keras.io/api/models/sequential/#sequential-class)

## Keras Layer

Keras [docs](https://keras.io/api/layers/base_layer/#layer-class)

## TensorFlow Module

TensorFlow [docs](https://www.tensorflow.org/api_docs/python/tf/Module)

## TensorFlow Tensor

TensorFlow [docs](https://www.tensorflow.org/api_docs/python/tf/Tensor)

## TensorFlow Variable

TensorFlow [docs](https://www.tensorflow.org/api_docs/python/tf/Variable)