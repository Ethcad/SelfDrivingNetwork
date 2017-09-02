#!/usr/bin/env python2

from sys import argv, exit
from os import listdir
from math import sqrt
from cv2 import imread
from scipy.misc import imresize
from keras.models import load_model
from numpy import float32, transpose, expand_dims
from PyQt5.QtWidgets import QLabel, QWidget, QApplication, QPushButton
from PyQt5.QtGui import QPixmap, QPalette, QImage, QTransform, QFont, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer


# Program which allows a user to visually compare and contrast a neural network and a human's steering abilities
# Created by brendon-ai, August 2017

class DataVisualizer(QWidget):

    # Degrees to rotate steering wheel, multiplied by the corresponding steering angle
    STEERING_WHEEL_COEFFICIENT = 360

    # Parameters for the low-pass filter
    LOW_PASS_VECTOR = [1.0, 0.5]

    # UI elements and counters
    video_display = None
    current_frame = 0
    num_frames = None
    frame_counter = None

    # Images and corresponding steering angles
    loaded_images = []
    actual_angles = []
    predicted_angles = []

    # Everything required for each of the two steering wheels and graph lines
    red_wheel = None
    red_wheel_label = None
    red_wheel_image = None
    red_line_points = []
    green_wheel = None
    green_wheel_label = None
    green_wheel_image = None
    green_line_points = []

    # Call various initialization functions
    def __init__(self):

        super(DataVisualizer, self).__init__()

        # Load images, set up the UI, and start updating
        standard_deviation = self.process_images()
        self.init_ui(standard_deviation)
        self.update_ui()

    # Initialize the user interface, with standard deviation parameter to be displayed on label
    def init_ui(self, standard_deviation):

        # Font used for the big labels under the steering wheels
        large_font = QFont('Source Sans Pro', 24)
        large_font.setWeight(30)

        # Font used for the steering angle indicators next to the graph
        small_font = QFont('Source Sans Pro', 16)
        small_font.setWeight(20)

        # Initialize a steering wheel and corresponding label at a given Y position
        def init_wheel_and_label(y):
            # Create a wheel and set its position
            wheel = QLabel(self)
            wheel.setAlignment(Qt.AlignCenter)
            wheel.setFixedSize(290, 288)
            wheel.move(1620, y)

            # Create a label, configure its text positioning and font, and set its position
            label = QLabel(self)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(290, 72)
            label.move(1620, y + 298)
            label.setFont(large_font)

            return wheel, label

        # Create a small graph indicator label with text and position generated from a steering angle
        def init_graph_label(steering_angle):
            y_point = self.get_line_graph_y_position(steering_angle) - 15
            label = QLabel(self)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(30, 30)
            label.move(1580, y_point)
            label.setFont(small_font)
            label.setText(str(steering_angle))

        # Black text on a light gray background
        palette = QPalette()
        palette.setColor(QPalette.Foreground, Qt.black)
        palette.setColor(QPalette.Background, Qt.lightGray)

        # Set the size, position, title, and color scheme of the window
        self.setFixedSize(1920, 860)
        self.move(0, 100)
        self.setWindowTitle('Training Data Visualizer')
        self.setPalette(palette)

        # Generate the buttons that handle skipping forward and backward
        skip_button_values = [-1000, -100, 100, 1000]
        for i in range(len(skip_button_values)):

            skip_value = skip_button_values[i]

            # Generate a function to skip forward or backward by frames
            def make_skip_function(frames):
                # Internal function for return
                def skip_frames():
                    self.current_frame += frames
                    self.red_line_points = []
                    self.green_line_points = []

                return skip_frames

            # Define the attributes of the button
            skip_button = QPushButton(self)
            skip_button.setFont(small_font)
            skip_button.setText("Skip %d" % skip_value)
            skip_button.clicked.connect(make_skip_function(skip_value))
            skip_button.setFixedSize(190, 40)

            # Calculate the X position of the button
            x_pos = (200 * i) + 10
            if skip_value > 0:
                x_pos += 800

            skip_button.move(x_pos, 810)

        # Initialize the label that shows the frame we are currently on
        self.frame_counter = QLabel(self)
        self.frame_counter.setAlignment(Qt.AlignCenter)
        self.frame_counter.setFont(small_font)
        self.frame_counter.setFixedSize(290, 40)
        self.frame_counter.move(1620, 810)

        # Initialize the label at the bottom that display low pass filter parameters and standard deviation
        std_dev_label = QLabel(self)
        std_dev_label.setAlignment(Qt.AlignCenter)
        std_dev_label.setFixedSize(800, 40)
        std_dev_label.move(410, 810)
        std_dev_label.setFont(small_font)
        std_dev_label.setText('Low pass filter parameters: %s              Standard deviation over whole run: %f'
                              % (self.LOW_PASS_VECTOR, standard_deviation))

        # Initialize the image box that holds the video frames
        self.video_display = QLabel(self)
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setFixedSize(1600, 528)
        self.video_display.move(10, 10)

        # Load the steering wheel images from the assets folder
        self.red_wheel_image = QImage("assets/red_wheel.png")
        self.green_wheel_image = QImage("assets/green_wheel.png")

        # Initialize the red and green wheels and corresponding labels
        self.red_wheel, self.red_wheel_label = init_wheel_and_label(10)
        self.green_wheel, self.green_wheel_label = init_wheel_and_label(435)

        # Create graph indicator labels for positions -0.1, 0.0, and 0.1
        init_graph_label(-0.1)
        init_graph_label(0.0)
        init_graph_label(0.1)

        # Make the window exist
        self.show()

    # Load all images from disk and calculate their human and network steering angles
    # Return standard deviation between human and network angles over whole run
    def process_images(self):

        # An implementation of a low pass filter, used to dampen the network's steering angle responses
        def low_pass_filter(steering_angle, output_memory):
            # Add current steering angle to memory of outputs (passed from outer function) and remove the oldest element
            output_memory = output_memory[:-1]
            output_memory.insert(0, steering_angle)

            # Weighted average of the last 5 outputs, using the low pass filter parameters as weights
            weighted_sum = sum(i[0] * i[1] for i in zip(output_memory, self.LOW_PASS_VECTOR))
            total_of_weights = sum(self.LOW_PASS_VECTOR)
            weighted_average = weighted_sum / total_of_weights

            return weighted_average, output_memory

        # Compute the standard deviation of the difference between two lists
        def standard_deviation(actual_output, ground_truth):
            total_squared_error = 0
            for i in range(len(actual_output)):
                error = actual_output[i] - ground_truth[i]
                squared_error = error * error
                total_squared_error += squared_error
            variance = total_squared_error / len(actual_output)
            return sqrt(variance)

        # If the arguments are invalid, fail to an error message
        try:
            # Load the Keras model from disk
            model = load_model(argv[1])

            # Load all image names from the folder, and record how many there are
            image_folder = argv[2]
            file_names = listdir(image_folder)
            image_names = [name for name in file_names if '.jpg' in name or '.png' in name]
            image_names.sort()
            self.num_frames = len(image_names)

            # Initialize a short-term memory of the last 5 raw outputs from the network, used for the low pass filter
            last_5_outputs = [0] * 5

            # Notify the user the process has begun; this may take a while
            print("Loading and processing images...")
            index = 0
            for image_name in image_names:
                # Take the human steering angle from the file name
                actual_angle = float(image_name.split("_")[1][:-4])
                self.actual_angles.append(actual_angle)

                # Load the image itself into a Numpy array
                image_path = ("%s/%s" % (image_folder, image_name))
                loaded_image = imread(image_path)
                self.loaded_images.append(loaded_image)

                # Pre-processing required for the network to classify it
                image_float = loaded_image.astype(float32)
                image_3d = transpose(image_float, (1, 0, 2))
                image_final = expand_dims(image_3d, 0)

                # Use the loaded model to predict a steering angle for the image, and apply a low pass filter
                predicted_angle = model.predict(image_final)
                (dampened_angle, last_5_outputs) = low_pass_filter(predicted_angle[0, 0], last_5_outputs)
                self.predicted_angles.append(dampened_angle)

                # Update the user every 1000 images
                index += 1
                if index % 1000 == 0:
                    print("Processed image %d of %d" % (index, self.num_frames))

            # Compute and return the standard deviation over the whole run
            return standard_deviation(self.predicted_angles, self.actual_angles)

        # Display an error message and quit
        except IndexError:
            print("Usage: ./data_visualizer.py <model> <image folder>")
            exit()

    # Updates the image box, steering wheels, and labels
    def update_ui(self):

        # Given a steering angle, rotate the steering wheel image and set the label correspondingly
        def set_wheel_angle(steering_angle, wheel_image, wheel, label, title):
            wheel_angle = steering_angle * self.STEERING_WHEEL_COEFFICIENT
            transform = QTransform().rotate(wheel_angle)
            pixmap = QPixmap.fromImage(wheel_image).transformed(transform)
            wheel.setPixmap(pixmap)
            label.setText("Steering angle\n(%s): %f" % (title, steering_angle))

        # Add a new point to the graph and shift all points left 5 pixels
        def update_point_list(point_list, steering_angle):
            y_point = self.get_line_graph_y_position(steering_angle)
            point_list.append([1570, y_point])
            for point in point_list:
                point[0] -= 5

        # Update the index that tells us what frame and steering angle to display
        image_index = self.current_frame % self.num_frames
        self.current_frame += 1

        # Update the label that displays the current frame
        self.frame_counter.setText("Frame %d / %d" % (image_index, self.num_frames))

        # Upscale a loaded image and display it
        frame = imresize(self.loaded_images[image_index], 4.0, interp='nearest')
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(image)
        self.video_display.setPixmap(pix)

        # Get the corresponding network and human steering angles for the current image
        red_angle = self.actual_angles[image_index]
        green_angle = self.predicted_angles[image_index]

        # Update the UI for the current steering angles
        set_wheel_angle(red_angle, self.red_wheel_image, self.red_wheel, self.red_wheel_label, "human")
        set_wheel_angle(green_angle, self.green_wheel_image, self.green_wheel, self.green_wheel_label, "network")

        # Add the current steering angles to the graph
        update_point_list(self.red_line_points, red_angle)
        update_point_list(self.green_line_points, green_angle)

        # Make sure the graph is redrawn every frame
        self.repaint()

        # Call this function again in 30 milliseconds
        QTimer().singleShot(30, self.update_ui)

    # Called when it is time to redraw
    def paintEvent(self, event):

        # Initialize the drawing tool
        painter = QPainter(self)

        # Draw a jagged line over a list of points
        def paint_line(point_list, color):
            # Configure the line color and width
            pen = QPen()
            pen.setColor(color)
            pen.setWidth(3)
            painter.setPen(pen)

            # Iterate over the points and draw a line between each consecutive pair
            previous_point = point_list[0]
            for i in range(1, len(point_list)):
                current_point = point_list[i]
                line_parameters = current_point + previous_point
                painter.drawLine(*line_parameters)
                previous_point = current_point

        # Calculate the Y points on the graph for steering angles of -0.1, 0.0, and 0.1 respectively
        y_0_1 = self.get_line_graph_y_position(-0.1)
        y0 = self.get_line_graph_y_position(0.0)
        y0_1 = self.get_line_graph_y_position(0.1)

        # Draw the three grid lines
        paint_line([[0, y_0_1], [1570, y_0_1]], QColor(0, 0, 0))
        paint_line([[0, y0], [1570, y0]], QColor(0, 0, 0))
        paint_line([[0, y0_1], [1570, y0_1]], QColor(0, 0, 0))

        # Draw the steering angle lines on the graph
        paint_line(self.red_line_points, QColor(255, 0, 0))
        paint_line(self.green_line_points, QColor(0, 255, 0))

    # Take an arbitrary steering angle, return the Y position that angle would correspond to on the graph
    @staticmethod
    def get_line_graph_y_position(steering_angle):
        y_point = -int(steering_angle * 800) + 669
        return y_point


# If this file is being run directly, instantiate the DataVisualizer class
if __name__ == '__main__':
    app = QApplication([])
    ic = DataVisualizer()
    exit(app.exec_())
