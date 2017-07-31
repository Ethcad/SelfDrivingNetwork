#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
import os

from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from skimage.io import imread
from sys import argv

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

model = load_model("./%s" % argv[1])
model = load_model("./%s" % argv[1])
fs_path = "/tmp/"
os.system("rm %s*sim*" % fs_path)
os.system("echo 0.0 > %s-1sim.txt" % fs_path)

i = 0
while True:
    path = "%ssim%d.png" % (fs_path, i)
    if os.path.isfile(path):
        image_raw = imread(path).astype(np.float32)
        image_3d = np.transpose(image_raw, (1, 0, 2))[:, 66:132, :]
        image = np.expand_dims(image_3d, 0)
        steering_angle = model.predict(image)
        os.system("echo %f > %stemp.txt" % (steering_angle, fs_path))
        os.system("mv %stemp.txt %s%dsim.txt" % (fs_path, fs_path, i))
        i += 1
