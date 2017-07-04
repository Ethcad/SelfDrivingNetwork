import tensorflow as tf
import numpy as np
import csv

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from skimage.io import imread

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    conv1 = tf.layers.conv2d(
        inputs=features,
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=2,
        strides=2
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=2,
        strides=2
    )

    pool2_flat = tf.reshape(pool2, [-1, 1382400])

    dense1 = tf.layers.dense(
        inputs=pool2_flat,
        units=256,
        activation=tf.nn.relu
    )

    dropout = tf.layers.dropout(
        inputs=dense1,
        rate=0.4,
        training=mode == learn.ModeKeys.TRAIN
    )

    dense2 = tf.layers.dense(
        inputs=dropout,
        units=1,
        activation=tf.nn.relu
    )

    predictions = tf.reshape(dense2, [-1])

    loss = None
    train_op = None

    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.mean_squared_error(
            labels=labels,
            predictions=predictions
        )

    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD"
        )

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )


def get_data():
    # Initialize arrays of data
    steering_angles = []
    image_list = []

    # Get image file paths and steering angles from CSV
    with open('data/labels.csv', 'r') as csv_file:
        filename_reader = csv.reader(csv_file)
        for row in filename_reader:
            steering_angles.append(float(row[1]))
            image_list.append(imread(row[0]))
    labels = np.array(steering_angles, dtype=np.float32)

    # Process and stack images
    image_count = len(image_list)
    image_size = image_list[0].shape
    training_images = np.empty((image_count,) + image_size, dtype=np.float32)
    for i in range(image_count):
        training_images[i, :, :, :] = image_list[i]

    return training_images, labels


def main(_):
    # Gather data
    images, labels = get_data()

    # Create the estimator
    classifier = learn.Estimator(
        model_fn=cnn_model_fn,
        model_dir="/tmp/network2"
    )

    # Train the model
    classifier.fit(
        x=images,
        y=labels,
        batch_size=10,
        steps=200
    )


if __name__ == "__main__":
    tf.app.run()
