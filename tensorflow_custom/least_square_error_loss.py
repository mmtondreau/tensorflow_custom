import tensorflow as tf


class LeaseSquareErrorLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(tf.convert_to_tensor(y_pred))
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
