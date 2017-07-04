from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv

from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    # Input layer
    input_layer = tf.reshape(features, [3, 720, 480, 1])

    # Convolutional layer
    conv = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )

    # Dense layer
    dense = tf.layers.dense(
        inputs=tf.reshape(conv, [-1, 3]),
        units=1
    )

    print(dense)

    # Calculate loss function (for both TRAIN and EVAL modes)
    loss = None
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.mean_squared_error(
            labels=labels,
            predictions=dense
        )

    # Configure the training op (for TRAIN mode)
    train_op = None
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.003,
            optimizer="SGD"
        )

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=dense,
        loss=loss,
        train_op=train_op
    )

def image_input_fn():
    # Initialize arrays of data
    file_paths = []
    steering_angles = []

    # Get image file paths and steering angles from CSV
    with open('data/labels.csv', 'r') as csvfile:
        filename_reader = csv.reader(csvfile)
        for row in filename_reader:
            file_paths.append(row[0])
            steering_angles.append(float(row[1]))

    labels = tf.convert_to_tensor(steering_angles, dtype=dtypes.float32)

    # Read images
    image_tensor = tf.convert_to_tensor(file_paths, dtype=dtypes.string)
    filename_queue = tf.train.string_input_producer(image_tensor)
    image_reader = tf.WholeFileReader()

    loaded_images = []
    for i in range(260):
        _, image_file = image_reader.read(filename_queue)
        decoded_image = tf.image.decode_png(image_file)
        features = tf.reshape(decoded_image, [3, -1])
        loaded_images.append(features)

    training_set = tf.cast(tf.stack(loaded_images, 2), tf.float32)

    return training_set, labels


def main(unused_argv):
    # Create the estimator
    classifier = learn.Estimator(
        model_fn=cnn_model_fn,
        model_dir="/tmp/cnn_simple_model"
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50
    )

    # Train the model
    classifier.fit(
        input_fn=image_input_fn,
        steps=260,
        monitors=[logging_hook]
    )

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy": learn.MetricSpec(
            metric_fn=tf.metrics.accuracy,
            prediction_key="classes"
        )
    }

    # # Evaluate the model and print results
    # eval_results = classifier.evaluate(
    #    x=eval_data,
    #    y=eval_labels,
    #    metrics=metrics
    # )


if __name__ == "__main__":
    tf.app.run()
