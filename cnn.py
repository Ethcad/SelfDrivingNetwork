import numpy as np
import csv

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from skimage.io import imread
from time import time


EPOCHS = 2
BATCH_SIZE = 32


def get_data(csv_path):
    # Initialize arrays of data
    steering_angles = []
    image_list = []

    # Get image file paths and steering angles from CSV
    with open(csv_path) as csv_file:
        filename_reader = csv.reader(csv_file)
        for row in filename_reader:
            steering_angles.append(float(row[1]) + 10)
            image_list.append(imread(row[0]))
    labels = np.array(steering_angles, dtype=np.float32)

    # Process and stack images
    image_count = len(image_list)
    image_size = image_list[0].shape
    training_images = np.empty((image_count,) + image_size, dtype=np.float32)
    for i in range(image_count):
        training_images[i, :, :, :] = image_list[i]

    return training_images, labels


def create_model():
    # Rectified linear activation function
    activation = 'relu'

    # Create the model
    model = Sequential()
    model.add(
        Conv2D(32, (2, 2),
               input_shape=(360, 240, 3),
               padding='same',
               activation=activation,
               #kernel_constraint=maxnorm(3)
               )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(64, (2, 2), padding='same'))
    model.add(Flatten())
    # model.add(Dense(128, activation=activation, kernel_constraint=maxnorm(3)))
    model.add(Dense(1, activation=activation))

    # Compile model
    learning_rate = 0.001
    decay = learning_rate / EPOCHS
    sgd = SGD(
        lr=learning_rate,
        momentum=0.9,
        decay=decay,
        nesterov=False
    )
    model.compile(
        loss='mean_squared_error',
        optimizer=sgd,
        metrics=[lambda y_true, y_pred: y_pred]
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

model.save("./model-%d.h5" % int(time()))
