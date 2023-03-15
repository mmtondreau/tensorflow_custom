import tensorflow as tf


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units: int, activation: str = None):
        super(LinearLayer, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.units),
            dtype="float32",
            trainable=True,
            initializer="random_normal",
        )

        self.b = self.add_weight(
            name="bias_range",
            shape=(self.units,),
            dtype="float32",
            trainable=True,
            initializer="random_normal",
        )

        self.c = self.add_weight(
            name="bias_domain",
            shape=(input_shape[-1],),
            dtype="float32",
            trainable=True,
            initializer="random_normal",
        )

    def call(self, inputs):
        return self.activation(tf.matmul(inputs + self.c, self.w) + self.b)
