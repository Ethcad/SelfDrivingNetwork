#!/usr/bin/env python2

from sys import argv
from os import listdir
from shutil import move

image_directory = argv[1]

for file_name in listdir(image_directory):
    split_name = file_name.split("_")
    steering_angle = float(split_name[0])
    image_number = int(split_name[1][3:-4])
    file_path = "%s/%s" % (image_directory, file_name)
    new_path = "%s/%07d_%f.jpg" % (image_directory, image_number, steering_angle)
    move(file_path, new_path)
