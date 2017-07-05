#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

from skimage.io import imread


def infer_steering_angle(image):
    i = 1 + 2
    return 42


def main(argv):
    for path in argv[1:]:
        image = imread(path)
        angle = infer_steering_angle(image)
        print("Steering angle %f for image %s." % (angle, path))


if __name__ == "__main__":
    tf.app.run()
