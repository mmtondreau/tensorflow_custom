import tensorflow as tf


class Huberloss(tf.keras.losses.Loss):
    def __init__(self, threshold=1):
        super().__init__()
        self.threshold = threshold

    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(tf.convert_to_tensor(y_pred))
        y_true = tf.cast(y_true, y_pred.dtype)
        threshold = tf.cast(self.threshold, dtype="float32")
        error = tf.subtract(y_pred, y_true)
        abs_error = tf.abs(error)
        half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)

        error = y_true - y_pred
        return tf.reduce_mean(
            tf.where(
                abs_error <= threshold,
                half * tf.square(error),
                threshold * abs_error - half * tf.square(threshold),
            ),
            axis=-1,
        )
