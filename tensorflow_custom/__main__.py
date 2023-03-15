import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_custom.training import fit
from tensorflow_custom.linear_layer import LinearLayer
from tensorflow_custom.hubber_loss import Huberloss
from tensorflow_custom.least_square_error_loss import LeaseSquareErrorLoss
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Check for TensorFlow GPU access
print(
    f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}"
)

# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")


NUM_EXAMPLES = 10000
BATCH_SIZE = 100
SHUFFLE_BUFFER_SIZE = 300
EPOCHS = 500

TRUE_w = 8.0
TRUE_b = 175.0


x = np.random.randint(-10000, 10000, size=(NUM_EXAMPLES,))
y = (TRUE_w * (x + 21)) + TRUE_b

# Split data into training and testing
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.2)


# shuffle and batch data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE).batch(
    batch_size=BATCH_SIZE
)
val_dataset = val_dataset.batch(batch_size=BATCH_SIZE)
test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)


def build_model():
    inputs = tf.keras.Input(shape=(1,), name="digits")
    x = tf.keras.layers.BatchNormalization()(inputs)
    outputs = LinearLayer(units=1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model = build_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
loss_object = LeaseSquareErrorLoss()
mse_metric = tf.keras.metrics.MeanSquaredError()
mae_metric = tf.keras.metrics.MeanAbsoluteError()


val_losses, train_losses = fit(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=EPOCHS,
    optimizer=optimizer,
    loss_object=loss_object,
    training_metrics=[mse_metric],
    validation_metrics=[mae_metric],
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            min_delta=0.05,
            baseline=None,
            mode="min",
            monitor="val_loss",
            restore_best_weights=True,
            verbose=1,
        )
    ],
)


def plot_metrics(train_metric, val_metric, metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.plot(train_metric, color="blue", label=metric_name)
    plt.plot(val_metric, color="green", label="val_" + metric_name)


# plot_metrics(train_losses, val_losses, "Loss", "Loss", ylim=1.0)
plt.plot(train_losses[10:], color="blue")
plt.plot(val_losses[10:], color="green")
plt.show()
print(train_losses)


def plot_data(inputs, outputs, predicted_outputs):
    real = plt.scatter(inputs, outputs, c="b", marker=".")
    predicted = plt.scatter(inputs, predicted_outputs, c="r", marker="+")
    plt.legend((real, predicted), ("Real Data", "Predicted Data"))
    plt.show()


plot_data(x_test, y_test, model(x_test))

print(f"mse: {loss_object(y_test, model(x_test))}")
