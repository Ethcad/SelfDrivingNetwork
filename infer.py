#!/usr/bin/env python2

import numpy as np
import os

from keras.models import load_model
from skimage.io import imread
from sys import argv

model = load_model("./%s" % argv[1])
os.system("rm /tmp/*sim* && echo 0.0 > /tmp/-1sim.txt")

i = 0
while True:
    path = "/tmp/sim%d.png" % i
    if os.path.isfile(path):
        image_raw = imread(path).astype(np.float32)
        image_3d = np.transpose(image_raw, (1, 0, 2))[:, 66:132, :]
        image = np.expand_dims(image_3d, 0)
        steering_angle = model.predict(image)
        os.system("echo %f > /tmp/temp.txt" % steering_angle)
        os.system("mv /tmp/temp.txt /tmp/%dsim.txt" % i)
        i += 1
