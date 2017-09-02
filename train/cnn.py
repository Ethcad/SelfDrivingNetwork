#!/usr/bin/env python2

import numpy as np

from skimage.io import imread
from time import time
from os import listdir, path, makedirs
from sys import argv
from math import tan, radians
from random import shuffle
from keras.callbacks import ModelCheckpoint

# Custom models
from mit_model import mit_model
from nvidia_model import nvidia_model
from more_filters_model import more_filters_model
from infinite_stack_model import infinite_stack_model


EPOCHS = 50
BATCH_SIZE = 1
VAL_PROPORTION = 0.1
START_TIME = int(time())


def get_data(file_path, should_convert, num_images):

    def consolidate_jagged_numpy_array(list_of_numpy_arrays):
        # Take a list of n-dimensional NumPy arrays and convert it to a single n+1-dimensional NumPy array
        image_count = len(list_of_numpy_arrays)
        image_size = list_of_numpy_arrays[0].shape
        consolidated_array = np.empty((image_count,) + image_size, dtype=np.float32)
        for i in range(image_count):
            consolidated_array[i, :, :, :] = list_of_numpy_arrays[i]

        return consolidated_array

    def steering_angle_to_inverse_turning_radius(steering_angle_degrees):
        # The wheel base of the car
        WHEEL_BASE = 0.914

        # Take a wheel angle in degrees and convert it to the inverse of the car's turning radius
        steering_angle_radians = radians(steering_angle_degrees)
        inverse_turning_radius = tan(steering_angle_radians) / WHEEL_BASE

        return inverse_turning_radius

    # Initialize arrays of data
    inverse_turning_radii = []
    image_list = []

    # Randomize list of files in directory and cut it to a length of num_images
    file_list = listdir(file_path)
    shuffle(file_list)
    if num_images > 0:
        file_list = file_list[:num_images]

    # Get image file paths and steering angles from folder
    i = 0
    num_files = len(file_list)
    for file_name in file_list:
        image_list.append(imread(file_path + file_name))
        file_name_end = file_name.split("_")[1]

        label_value = float(file_name_end[:-4])
        if should_convert:
            label_value = steering_angle_to_inverse_turning_radius(label_value)
        inverse_turning_radii.append(label_value)

        i += 1
        if i % 1000 == 0:
            print("Loaded %d of %d images from directory '%s'." % (i, num_files, file_path))

    angle_labels = np.array(inverse_turning_radii, dtype=np.float32)

    # Consolidate lists of numpy arrays, split into val and train
    val_num = int(len(angle_labels) * VAL_PROPORTION)

    image_list_train = image_list[:-val_num]
    training_images = consolidate_jagged_numpy_array(image_list_train)
    image_list_val = image_list[-val_num:]
    validation_images = consolidate_jagged_numpy_array(image_list_val)

    angle_labels_train = angle_labels[:-val_num]
    angle_labels_val = angle_labels[-val_num:]

    return training_images, angle_labels_train, validation_images, angle_labels_val


def should_convert_data_set():
    # Check for command line flag '--steering-angle'
    should_convert = False
    for argument in argv:
        if argument == "--steering-angle":
            should_convert = True

    if should_convert:
        print("Converting labels to inverse turning radii.")
    else:
        print("Not converting labels.")
    return should_convert

# Create a directory for snapshots to be saved in
makedirs("../model/%d" % START_TIME)

# Select the number of images to cut out of the data set
num_images = 0
if len(argv) >= 3:
    num_images = int(argv[2])

# Gather images and reorder their dimensions
axis_order = (0, 2, 1, 3)
images, labels, images_val, labels_val = get_data(argv[1], should_convert_data_set(), num_images)
images_t = np.transpose(images, axis_order)
images_val_t = np.transpose(images_val, axis_order)

# Initialize list of models
models = [
#    infinite_stack_model(4, 2),
#    nvidia_model(),
#    mit_model(),
#    more_filters_model([
#        (64, 1, True),
#        (128, 3, True)
#    ]),
#    more_filters_model([
#        (64, 2, True),
#        (256, 3, False)
#    ])
#    more_filters_model([
#        (64, 2, True),
#        (128, 3, False)
#    ],
#    320, 180)
    nvidia_model()
]

# Train each network with the same set of images, save the trained model, and print results
for i in range(len(models)):
    print("\nSummary of model #%d:" % i)
    print(models[i].summary())

    models[i].fit(
        images_t,
        labels,
        validation_data=(images_val_t, labels_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[ModelCheckpoint("../model/%d/epoch={epoch:02d}-val_loss={val_loss:f}.h5" % START_TIME)]
    )
