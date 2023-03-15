import tensorflow as tf

import numpy as np
from tqdm import tqdm


def apply_gradient(optimizer, loss_object, model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss_object(y_true=y, y_pred=logits)

    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return logits, loss_value


def train_data_for_one_epoch(
    train_dataset,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_object: tf.keras.losses.Loss,
    model: tf.keras.Model,
    metrics: list[tf.keras.metrics.Metric],
    callbacks: tf.keras.callbacks.CallbackList,
):
    losses = []
    pbar = tqdm(
        total=len(list(enumerate(train_dataset))),
        position=0,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ",
    )
    batch_logs = {}
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        callbacks.on_train_batch_begin(step)
        logits, loss = apply_gradient(
            optimizer, loss_object, model, x_batch_train, y_batch_train
        )

        losses.append(loss)
        batch_logs["loss"] = loss
        for metric in metrics:
            metric.update_state(y_batch_train, logits)
            batch_logs[metric.name] = metric.result()
        callbacks.on_train_batch_end(step, batch_logs)

        pbar.set_description(
            "Training loss for step %s: %.4f" % (int(step), float(loss))
        )
        pbar.update()
        if model.stop_training:
            break
    return np.mean(losses)


def perform_validation(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    loss_object: tf.keras.losses.Loss,
    metrics: tf.keras.metrics.Metric,
):
    losses = []
    for x_val, y_val in test_dataset:
        val_logits = model(x_val)
        val_loss = loss_object(y_true=y_val, y_pred=val_logits)
        losses.append(val_loss)
        for metric in metrics:
            metric.update_state(y_val, val_logits)
    return np.mean(losses)


def fit(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_object: tf.keras.losses.Loss,
    training_metrics: list[tf.keras.metrics.Metric],
    validation_metrics: list[tf.keras.metrics.Metric],
    callbacks: list[tf.keras.callbacks.Callback],
):

    epochs_val_losses, epochs_train_losses = [], []
    callbacks = tf.keras.callbacks.CallbackList(callbacks, model=model)
    callbacks.on_train_begin()
    for epoch in range(epochs):
        # Reset states of all metrics
        for metric in training_metrics + validation_metrics:
            metric.reset_states()

        callbacks.on_epoch_begin(epoch)
        print("Start of epoch %d" % (epoch,))
        epoch_logs = {}

        # Perform Training over all batches of train data
        epoch_logs["loss"] = train_data_for_one_epoch(
            train_dataset,
            optimizer,
            loss_object,
            model,
            training_metrics,
            callbacks=callbacks,
        )

        # Get results from training metrics
        for training_metric in training_metrics:
            epoch_logs[training_metric.name] = training_metric.result()

        # Perform validation on all batches of test data
        epoch_logs["val_loss"] = perform_validation(
            model, val_dataset, loss_object, validation_metrics
        )

        # Get results from validation metrics
        for validation_metric in validation_metrics:
            epoch_logs[f"val_{training_metric.name}"] = validation_metric.result()
        callbacks.on_epoch_end(epoch, epoch_logs)

        # Calculate training and validation losses for current epoch
        losses_train_mean = epoch_logs["loss"]
        losses_val_mean = epoch_logs["val_loss"]
        epochs_val_losses.append(losses_val_mean)
        epochs_train_losses.append(losses_train_mean)

        # Output metrics
        metric_output = ", ".join(
            [f"{key}: {value}" for key, value in epoch_logs.items()]
        )
        loss_output = f"\n Epcoh {epoch}:"
        print(",".join([loss_output, metric_output]))
        if model.stop_training:
            break
    # process final logs
    final_logs = {}
    for training_metric in training_metrics:
        final_logs[training_metric.name] = training_metric.result()
    for validation_metric in validation_metrics:
        final_logs[f"val_{validation_metric.name}"] = validation_metric.result()
    callbacks.on_train_end(final_logs)

    return epochs_val_losses, epochs_train_losses
