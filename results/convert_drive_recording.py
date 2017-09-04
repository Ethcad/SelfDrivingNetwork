#!/usr/bin/env python2

from sys import argv
from os import listdir
from shutil import move

# Rename the images in the directory passed as a command line argument
image_directory = argv[1]

# For every file in the directory:
for file_name in listdir(image_directory):
    # Extract the number from the file name
    image_number = int(file_name[3:-4])
    # Generate a new name with a steering angle of zero
    new_name = "%s/0.0_%d.jpg" % (image_directory, image_number)
    # Move the image
    move("%s/%s" % (image_directory, file_name), new_name)
