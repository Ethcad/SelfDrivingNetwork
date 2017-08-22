#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
import os

from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from skimage.io import imread
from sys import argv

# Path to look for images in and record classifications in
TEMP_PATH = "/tmp/"

# Limit the amount of GPU memory the TensorFlow backend will use; out-of-memory error occurs otherwise on the Jetson
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# Load the model twice; for some reason it fails to load correctly the first time on the Jetson
model = load_model("./%s" % argv[1])
model = load_model("./%s" % argv[1])

# Clear old data from the temp folder and record an initial output
os.system("rm %s*sim*" % TEMP_PATH)
os.system("echo 0.0 > %s-1sim.txt" % TEMP_PATH)

# Loop forever, classifying images and recording outputs to files
i = 0
while True:
    # Read from last image plus one (there should not be any gaps)
    path = "%ssim%d.jpg" % (TEMP_PATH, i)
    if os.path.isfile(path):
        # Read the file as a 32-bit floating point tensor
        image_raw = imread(path).astype(np.float32)

        # Rearrange and crop it into a format that the neural network should accept
        image_3d = np.transpose(image_raw, (1, 0, 2))[60:260, 96:162, :]

        # Add an extra dimension (used for batch stacking by Keras)
        image = np.expand_dims(image_3d, 0)

        # Classify the image
        steering_angle = model.predict(image)

        # Write the classification to a temp file and rename it
        os.system("echo %f > %stemp.txt" % (steering_angle, TEMP_PATH))
        os.system("mv %stemp.txt %s%dsim.txt" % (TEMP_PATH, TEMP_PATH, i))

        # Increment the image counter
        i += 1
