#!/usr/bin/env python2

from sys import argv
from os import listdir
from shutil import move

# Rename the images in the directory passed as a command line argument
image_directory = argv[1]

# For every file in the directory:
for file_name in listdir(image_directory):
    # Split the name at the underscore
    split_name = file_name.split("_")
    # The first part is the steering angle
    steering_angle = float(split_name[0])
    # The second part is the unique (within that dataset) identifier
    image_number = int(split_name[1][3:-4])
    # Shift all of the numbers in the dataset up by an arbitrary number
    new_number = image_number + int(argv[2])
    # Format the new name and move the image
    file_path = "%s/%s" % (image_directory, file_name)
    new_path = "%s/%07d_%f.jpg" % (image_directory, new_number, steering_angle)
    move(file_path, new_path)
