#!/usr/bin/env python2

from sys import argv, exit
from os import listdir
from cv2 import imread
from scipy.misc import imresize
from threading import Thread, Event
from keras.models import load_model
from numpy import float32, transpose, expand_dims
from PyQt5.QtWidgets import QLabel, QWidget, QApplication
from PyQt5.QtGui import QPixmap, QPalette, QImage, QTransform, QFont
from PyQt5.QtCore import Qt


class TimerThread(Thread):

    data_visualizer = None

    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def run(self):
        while not self.stopped.wait(0.03):
            self.data_visualizer.update_ui()


class DataVisualizer(QWidget):

    ANGLE_COEFFICIENT = 360

    video_display = None
    current_frame = 0
    num_frames = None
    loaded_images = []
    actual_angles = []
    predicted_angles = []
    red_wheel = None
    red_wheel_label = None
    red_wheel_image = None
    green_wheel = None
    green_wheel_label = None
    green_wheel_image = None

    def __init__(self):

        super(DataVisualizer, self).__init__()

        self.process_images()
        self.init_ui()

        stop_flag = Event()
        timer_thread = TimerThread(stop_flag)
        timer_thread.daemon = True
        timer_thread.start()
        timer_thread.data_visualizer = self

    def init_ui(self):

        font = QFont('Source Sans Pro', 24)
        font.setWeight(30)

        def init_wheel_and_label(y):
            wheel = QLabel(self)
            wheel.setAlignment(Qt.AlignCenter)
            wheel.setFixedSize(290, 288)
            wheel.move(1620, y)

            label = QLabel(self)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(290, 72)
            label.move(1620, y + 298)
            label.setFont(font)

            return wheel, label

        palette = QPalette()
        palette.setColor(QPalette.Foreground, Qt.black)
        palette.setColor(QPalette.Background, Qt.lightGray)

        self.setFixedSize(1920, 800)
        self.move(0, 100)
        self.setWindowTitle('Training Data Visualizer')
        self.setPalette(palette)

        self.video_display = QLabel(self)
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setFixedSize(1600, 528)
        self.video_display.move(10, 10)

        self.red_wheel_image = QImage("red_wheel.png")
        self.green_wheel_image = QImage("green_wheel.png")

        self.red_wheel, self.red_wheel_label = init_wheel_and_label(10)
        self.green_wheel, self.green_wheel_label = init_wheel_and_label(410)

        self.show()

    def process_images(self):

        try:
            model = load_model(argv[1])
            image_folder = argv[2]
            file_names = listdir(image_folder)
            image_names = [name for name in file_names if '.jpg' in name or '.png' in name]
            image_names.sort()
            self.num_frames = len(image_names)

            print "Loading and processing images..."
            index = 0
            for image_name in image_names:
                actual_angle = float(image_name.split("_")[1][:-4])
                self.actual_angles.append(actual_angle)

                image_path = ("%s/%s" % (image_folder, image_name))
                loaded_image = imread(image_path)
                self.loaded_images.append(loaded_image)

                image_float = loaded_image.astype(float32)
                image_3d = transpose(image_float, (1, 0, 2))
                image_final = expand_dims(image_3d, 0)

                predicted_angle = model.predict(image_final)
                self.predicted_angles.append(predicted_angle)

                index += 1
                if index % 1000 == 0:
                    print "Processed image %d of %d" % (index, self.num_frames)

        except IndexError:
            print "Usage: ./data_visualizer.py <model> <image folder>"
            exit()

    def update_ui(self):

        def set_wheel_angle(steering_angle, wheel_image, wheel, label, title):
            wheel_angle = steering_angle * self.ANGLE_COEFFICIENT
            transform = QTransform().rotate(wheel_angle)
            pixmap = QPixmap.fromImage(wheel_image).transformed(transform)
            wheel.setPixmap(pixmap)
            label.setText("Steering angle\n(%s): %f" % (title, steering_angle))

        image_index = self.current_frame % self.num_frames
        self.current_frame += 1
        frame = imresize(self.loaded_images[image_index], 8.0, interp='nearest')
        # frame = imresize(imread("example.png"), 8.0, interp='nearest')
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(image)
        self.video_display.setPixmap(pix)

        set_wheel_angle(self.actual_angles[image_index], self.red_wheel_image, self.red_wheel, self.red_wheel_label, "human")
        set_wheel_angle(self.predicted_angles[image_index], self.green_wheel_image, self.green_wheel, self.green_wheel_label, "network")


if __name__ == '__main__':
    app = QApplication([])
    ic = DataVisualizer()
    exit(app.exec_())
