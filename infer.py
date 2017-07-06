#!/usr/bin/env python2

import numpy as np

from keras.models import load_model
from skimage.io import imread
from sys import argv

model = load_model("./%s" % argv[1])

for path in argv[2:]:
    image_raw = imread(path).astype(np.float32)
    image_3d = np.transpose(image_raw, (1, 0, 2))
    image = np.expand_dims(image_3d, 0)
    steering_angle = model.predict(image)
    print(steering_angle)
