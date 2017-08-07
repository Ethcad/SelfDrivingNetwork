#!/usr/bin/env bash

gst-launch-1.0 -v v4l2src device=/dev/video1 \
! image/jpeg, width=1920, height=1080, framerate=30/1 \
! jpegparse ! multifilesink location="sim%d.jpg"
