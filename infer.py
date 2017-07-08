#!/usr/bin/env python2

import numpy as np
import os

from keras.models import load_model
from skimage.io import imread
from sys import argv

model = load_model("./%s" % argv[1])
os.system("rm /tmp/sim* && echo 0.0 > /tmp/sim")

i = 1
while True:
    if os.path.isfile("/tmp/sim%d.png" % i):
        image_raw = imread("/tmp/sim%d.png" % (i - 1)).astype(np.float32)
        image_3d = np.transpose(image_raw, (1, 0, 2))
        image = np.expand_dims(image_3d, 0)
        steering_angle = model.predict(image)
        os.system("echo %f > /tmp/sim" % steering_angle)
        i += 1
