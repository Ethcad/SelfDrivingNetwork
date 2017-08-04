import numpy as np

from skimage.io import imread
from time import time
from os import listdir

# Custom models
from mit_model import mit_model
from nvidia_model import nvidia_model


EPOCHS = 5
BATCH_SIZE = 10
VAL_PROPORTION = 0.1


def consolidate_jagged_numpy_array(list_of_numpy_arrays):
    # Process and stack images
    image_count = len(list_of_numpy_arrays)
    image_size = list_of_numpy_arrays[0].shape
    consolidated_array = np.empty((image_count,) + image_size, dtype=np.float32)
    for i in range(image_count):
        consolidated_array[i, :, :, :] = list_of_numpy_arrays[i]

    return consolidated_array


def get_data(path):
    # Initialize arrays of data
    steering_angles = []
    image_list = []

    # Get image file paths and steering angles from folder
    i = 0
    file_list = listdir(path)
    num_files = len(file_list)
    for file_name in file_list:
        image_list.append(imread(path + file_name))
        file_name_end = file_name.split("_")[1]
        steering_angles.append(float(file_name_end[:-4]))
        i += 1
        if i % 1000 == 0:
            print("Loaded %d of %d images from directory '%s'." % (i, num_files, path))

    angle_labels = np.array(steering_angles, dtype=np.float32)

    # Consolidate lists of numpy arrays, split into val and train
    val_num = int(len(angle_labels) * VAL_PROPORTION)

    image_list_train = image_list[:-val_num]
    training_images = consolidate_jagged_numpy_array(image_list_train)
    image_list_val = image_list[-val_num:]
    validation_images = consolidate_jagged_numpy_array(image_list_val)

    angle_labels_train = angle_labels[:-val_num]
    angle_labels_val = angle_labels[-val_num:]

    return training_images, angle_labels_train, validation_images, angle_labels_val


# Gather data
axis_order = (0, 2, 1, 3)
images, labels, images_val, labels_val = get_data("./data/")
images_t = np.transpose(images, axis_order)
images_val_t = np.transpose(images_val, axis_order)

model = nvidia_model()

model.fit(
    images_t,
    labels,
    validation_data=(images_val_t, labels_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.save("./model/%d.h5" % int(time()))
