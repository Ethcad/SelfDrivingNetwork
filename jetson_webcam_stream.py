import cv2
from time import sleep

cam_capture = cv2.VideoCapture(1)
cam_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
cam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
cam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 150)

i = 0
while True:
    frame_raw = cam_capture.read()[1]
    frame = frame_raw[66:132, :]
    cv2.imwrite("%d.png" % i, frame)
    sleep(0.02)
    i += 1
