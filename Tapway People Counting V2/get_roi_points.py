# import the necessary packages
import argparse
import cv2
from PIL import Image
import numpy as np
import json

# Read JSON Configuration File #
with open('config.json') as f:
    data = json.load(f)

# Input camera's name #
camera_name = input("Enter your camera name: ")

for item in data['cameras']:
	if item['camera_name'] == camera_name:
		cam = item
		input = item['input_filename']
		width = item['output_video_width']
		height = item['output_video_height']
		break

tables = cam["p"]

cap = cv2.VideoCapture(input) 
flag, frame = cap.read()
cimage = Image.fromarray(frame)
cimage.save('frame.png')
image = cv2.imread('frame.png')
image = cv2.resize(image, (width, height))

# load the image, clone it, and setup the mouse callback function
cv2.namedWindow("People Tracking - ROI Selector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("People Tracking - ROI Selector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# keep looping until the 'q' key is pressed
while True:
	# Draw table box, (x, y, z) is the RGB value to change color
	for num in tables: 
		p1 = (num['x1'], num['y1'])
		p2 = (num['x2'], num['y2'])	
		cv2.rectangle(image, p1, p2, (204, 0, 102), 1)

	# display the image and wait for a keypress
	cv2.imshow("People Tracking - ROI Selector", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, start selecting an area to get the ROI points!
	# After done, press ENTER to get the values!
	if key == ord("r"):
		r = cv2.selectROI("People Tracking - ROI Selector", image, fromCenter=False, showCrosshair=True)
		print("x1 = " + str(r[0]))
		print("y1 = " + str(r[1]))
		print("x2 = " + str(r[0]+r[2]))
		print("y2 = " + str(r[1]+r[3]))
		#new_ROI = {'table_id': int(cam['num_of_tables']), "x1": int(r[0]), "y1": int(r[1]), "x2": int(r[0]+r[2]), "y2": int(r[1]+r[3])}
	
	# if the 'q' key is pressed, do not save ROI, but the result is shown on command prompt!
	elif key == ord("q"):
		print("The program will quit now.")
		break


# close all open windows
cv2.destroyAllWindows()