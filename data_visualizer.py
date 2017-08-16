from sys import argv, exit
from cv2 import imread
from scipy.misc import imresize
from threading import Thread, Event
from PyQt5.QtWidgets import QLabel, QWidget, QApplication
from PyQt5.QtGui import QPixmap, QPalette, QImage
from PyQt5.QtCore import Qt


class TimerThread(Thread):

    data_visualizer = None

    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def run(self):
        while not self.stopped.wait(0.02):
            self.data_visualizer.update_ui()


class DataVisualizer(QWidget):

    WINDOW_WIDTH = 1600
    WINDOW_HEIGHT = 528

    video_frame = None

    def __init__(self):
        super(DataVisualizer, self).__init__()

        self.init_ui()

        stop_flag = Event()
        timer_thread = TimerThread(stop_flag)
        timer_thread.daemon = True
        timer_thread.start()
        timer_thread.data_visualizer = self

    def init_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.Foreground, Qt.white)
        palette.setColor(QPalette.Background, Qt.darkGray)

        self.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.move(100, 100)
        self.setWindowTitle('Training Data Visualizer')
        self.setPalette(palette)

        # font = QFont('Source Sans Pro', 30)
        # title = QLabel(self)
        # title.setAlignment(Qt.AlignCenter)
        # title.setFixedSize(1280, 60)
        # title.move(0, 10)
        # title.setText('Live Fruit Classifier')
        # title.setPalette(palette)
        # title.setFont(font)

        self.video_frame = QLabel(self)
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.video_frame.move(0, 0)

        self.show()

    def update_ui(self):
        frame_small = imread("example.png")
        frame = imresize(frame_small, 8.0, interp='nearest')
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(image)
        self.video_frame.setPixmap(pix)


if __name__ == '__main__':
    app = QApplication(argv)
    ic = DataVisualizer()
    exit(app.exec_())
