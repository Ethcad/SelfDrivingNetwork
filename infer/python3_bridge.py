#!/usr/bin/env python3

import os
import sys
import numpy as np

from time import sleep
from keras.models import load_model
from cv2 import imread
from scipy.ndimage.interpolation import zoom
from LaneDetection.infer.steering_engine import SteeringEngine
from LaneDetection.infer.sliding_window_inference_engine import SlidingWindowInferenceEngine

# Verify that the number of command line arguments is correct
if len(sys.argv) != 3:
    print('Usage: {} <model 1> <model 2>'.format(sys.argv[0]))
    sys.exit()

# List of two inference engines, one for each line
inference_engines = []

# For each of the two models passed as command line arguments
for arg in sys.argv[1:]:

    # Format the fully qualified path of the trained model
    model_path = os.path.expanduser(arg)

    print(model_path)
    # Load the model from disk
    model = load_model(model_path)

    # Create an inference engine
    inference_engine = SlidingWindowInferenceEngine(
        model=model,
        window_size=16,
        stride=8
    )

    # Add the inference engine to the list
    inference_engines.append(inference_engine)

# Create a steering engine
steering_engine = SteeringEngine(
    max_average_variation=20,
    steering_multiplier=0.1,
    ideal_center_x=160,
    steering_limit=0.2
)

# Notify the webcam streaming program that we have finished loading
print('ready')

# Loop forever, waiting for input on each iteration
while True:

    # Contains the path of the image
    image_path = None

    # Get the fully qualified file path of an image from a temp file
    with open('/tmp/drive.path') as path_file:
        image_path = os.path.expanduser(path_file.read().strip())

    # Try to read the file from disk as a 32-bit floating point tensor
    print(image_path)
    image_raw = imread(image_path).astype(np.float32)

    # Resize the image, rearrange the dimensions and add an extra one (used for batch stacking by Keras)
    image_transpose = np.transpose(image_raw, (1, 0, 2))
    image_cropped = image_transpose[:,40:-80, :]

    # List containing yellow line and white line
    lines = []

    # With each inference engine
    for inference_engine in inference_engines:

        # Find points on the line that the current engine is trained to detect
        line = inference_engine.infer(image_cropped)

        # Add the current line to the list of lines
        lines.append(line)

    # Calculate a steering angle from the lines with the steering engine
    steering_angle = steering_engine.compute_steering_angle(*lines)

    # Write the angle to a file
    with open('/tmp/drive.angle', 'w') as angle_file:
        angle_file.write(str(steering_angle))
    print('test')

