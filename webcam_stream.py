#!/usr/bin/env python2

from os import system, popen
from pexpect import pxssh

system('gst-launch-1.0 -v v4l2src device=/dev/video1 ! image/jpeg, width=320, height=180, framerate=30/1 ! jpegparse ! multifilesink location="/tmp/sim%d.jpg" &')

s = pxssh.pxssh()
s.login('192.168.0.250', 'admin', '')
while True:
	s.sendline('cat /home/lvuser/latest.encval')
	s.prompt()
	last_max_file = -1
	for line in iter(s.before.splitlines()):
		if 'out' in line:
			encoder_value = int(line[3:])
			print encoder_value
			max_file = last_max_file
			for file_name in popen('ls -1 /tmp/sim*.jpg').read().split('\n')[:-1]:
				file_number = int(file_name[8:-4])
				if file_number > max_file:
					max_file = file_number
			if max_file > -1:
				system('mv /tmp/sim%d.jpg /tmp/%d_sim%d.jpg' % (max_file, encoder_value, max_file))
				last_max_file = max_file
