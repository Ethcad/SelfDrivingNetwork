#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn
from skimage.io import imread
from cnn import cnn_model_fn

tf.logging.set_verbosity(tf.logging.FATAL)


def infer_steering_angle(classifier, image):
    output = classifier.predict(
        x=image,
        batch_size=1
    )
    for angle in output:
        return angle


def import_model():
    # Load estimator
    classifier = learn.Estimator(
        model_fn=cnn_model_fn,
        model_dir="/tmp/network2"
    )
    return classifier


def main(argv):
    classifier = import_model()
    for path in argv[1:]:
        image_reversed = imread(path).astype(np.float32)
        image_unlayered = np.transpose(image_reversed, (1, 0, 2))
        image = np.reshape(image_unlayered, [1, -1, 480, 3])
        angle = infer_steering_angle(classifier, image)
        print("Steering angle %f for image %s." % (angle, path))


if __name__ == "__main__":
    tf.app.run()
