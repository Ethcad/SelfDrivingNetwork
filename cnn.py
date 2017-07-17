import numpy as np
import csv

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adadelta
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from skimage.io import imread
from matplotlib.pyplot import imshow
from time import time


EPOCHS = 5
BATCH_SIZE = 4


def get_data(csv_path):
    # Initialize arrays of data
    steering_angles = []
    image_list = []

    # Get image file paths and steering angles from CSV
    with open(csv_path) as csv_file:
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
        if i % 200 == 0:
            print("Loaded image %i of %i from file '%s'" % (i, image_count, csv_path))
    imshow(image_list[1])

    return training_images, labels


def create_model():
    # Rectified linear activation function
    activation = 'tanh'

    # Create the model
    model = Sequential()
    model.add(Conv2D(
        input_shape=(200, 66, 3 ),
        filters=8,
        strides=(3, 3),
        kernel_size=(3, 3),
        padding='same',
        activation=activation,
    ))
    for i in range(1):
        model.add(Conv2D(
            filters=8,
            strides=(3, 3),
            kernel_size=(3, 3),
            padding='same',
            activation=activation,
        ))
    model.add(MaxPooling2D(
        pool_size=(4, 4),
        strides=(2, 2)
    ))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dense(1))

    # Compile model
    optimizer = Adadelta()
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )

    print(model.summary())
    return model


# Gather data
axis_order = (0, 2, 1, 3)
images, labels = get_data("./data/labels.csv")
images_t = np.transpose(images, axis_order)
images_val, labels_val = get_data("./data/labels_val.csv")
images_val_t = np.transpose(images_val, axis_order)

model = create_model()

model.fit(
    images_t,
    labels,
    validation_data=(images_val_t, labels_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.save("./model/%d.h5" % int(time()))
