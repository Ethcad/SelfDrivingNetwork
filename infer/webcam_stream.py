#!/usr/bin/env python2

from os import system, popen, listdir
from sys import argv
from time import time
from libssh2 import Session
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread
from evdev import InputDevice, categorize, ecodes, KeyEvent

recording_encoder = False


# Parallel thread that accepts joystick input and sets a global flag
def handle_gamepad_input():
    global recording_encoder
    
    # Get the input from the device file (specific to the joystick I am using)
    joystick = InputDevice('/dev/input/by-id/usb-Logitech_Logitech_Dual_Action_E89BB55E-event-joystick')
    
    # A provided loop that will run forever, iterating on inputs as they come
    for event in joystick.read_loop():
        
        # Is the input a button or key press?
        if event.type == ecodes.EV_KEY:
            # Get the identifier of the button that was pressed
            key_event = categorize(event)

            # Set the global flag to true if the A button was pressed, false if B was pressed
            if key_event.keycode == "BTN_THUMB":
                recording_encoder = True
            elif key_event.keycode == "BTN_THUMB2":
                recording_encoder = False

# Save images in folder provided as a command line argument
image_folder = argv[1]

# Configure the webcam
system('v4l2-ctl -d /dev/video1 --set-ctrl=exposure_auto=3')
system('v4l2-ctl -d /dev/video1 --set-ctrl=exposure_auto_priority=1')
system('v4l2-ctl -d /dev/video1 --set-ctrl=exposure_absolute=250')

# Start the camera capture daemon process from the command line
system('gst-launch-1.0 -v v4l2src device=/dev/video1 ! image/jpeg, width=320, height=180, framerate=30/1 ! jpegparse ! multifilesink location="%s/sim%d.jpg" &' % image_folder)

# Open an SSH session to the robot controller
sock = socket(AF_INET, SOCK_STREAM)
sock.connect(('192.168.0.230', 22))

session = Session()
session.startup(sock)
session.userauth_password('admin', '')

channel = session.open_session()
channel.shell()

# Start the joystick input thread
thread = Thread(target=handle_gamepad_input)
thread.daemon = True
thread.start()

# Loop forever, recording steering angle data along with images
i = 0
while True:
    i += 1

    # Compose values for transfer to robot controller into a single string
    values_to_jetson = (int(recording_encoder), 0, i)
    values_str = ''
    for value in values_to_jetson:
        values_str += (str(value) + '\n')
    values_str = values_str[:-1]

    # Send values over SSH to the robot controller by writing them to a temp file and then renaming it
    channel.write('printf "%s" > /home/lvuser/temp.txt\n' % values_str)
    channel.write('mv /home/lvuser/temp.txt /home/lvuser/values.txt\n')

    # To be executed if we are supposed to be recording steering angle data currently
    if recording_encoder:
        # Prompt robot controller to send us current encoder position
        channel.write('cat /home/lvuser/latest.encval\n')
        stdout = channel.read(1024)

        # Loop over SSH output
        last_max_file = -1
        for line in stdout.split("\n"):
            # Only one iteration should occur each time because the input should have just one line containing 'out'
            if 'out' in line:
                # Extract the encoder position from the line
                encoder_value = float(line[3:])

                # Loop over all camera capture images currently in the temp directory
                # Tracking of the last file with the highest number ensures no duplicate data is recorded
                max_file = last_max_file
                for file_name in listdir(image_folder):
                    if "sim" in file_name and ".jpg" in file_name and "_" not in file_name:
                        # Extract the Unix timestamp from the file name
                        file_number = int(file_name[3:-4])
                        # Find the newest image file
                        if file_number > max_file:
                            max_file = file_number

                # If a new value has been obtained, record it in the file name of the latest image
                if max_file > last_max_file:
                    system('mv %s/sim%d.jpg %s/%f_sim%d.jpg' % (image_folder, max_file, image_folder, encoder_value, max_file))

                # Set the previous image counter to the current image's timestamp
                last_max_file = max_file
