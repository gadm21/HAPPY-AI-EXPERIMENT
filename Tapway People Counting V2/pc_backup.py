# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

# Import Libraries #
import argparse
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
import database.db_info as db
import track.tracker as track
import detector as od
import json

# Read JSON Configuration File #
with open('config.json') as f:
    data = json.load(f)

# Read from 1 camera: cam0 #
# Later on, add support for multiple camera here with threading if needed #
for camera in data['cameras']:
	if camera['camera_name'] == "cam0": cam0 = camera
tables = cam0["p"]

# Create track for all the tables in the camera
tracker = []
for i in range(cam0['num_of_tables']):
	tracker.append(track.Tracker())

# MYSQL Database Connection #
mydb, db_cursor = db.db_connection()
db_cursor.execute("SELECT MAX(ID) FROM people")
id_query = db_cursor.fetchone()

if id_query['MAX(ID)'] == None:
	id = 0
else:
	id = id_query['MAX(ID)']
	

# Function to draw boxes around people #
def drawTrackedPeople(imgDisplay):
	#print(tracker.faceTrackers.keys()) # DEBUG: display person id in session list
	for index in range(cam0['num_of_tables']):
		for fid in tracker[index].faceTrackers.keys():
			#print(fid) # DEBUG: display current person id in processing
			tracked_position = tracker[index].faceTrackers[fid].get_position()
			t_x = int(tracked_position.left())
			t_y = int(tracked_position.top())
			t_w = int(tracked_position.width())
			t_h = int(tracked_position.height())

			timestayed = int(tracker[index].getTimeInSecond(fid))
			
			db_cursor.execute( "SELECT ID FROM people" )
			peoples = [item['ID'] for item in db_cursor.fetchall()]
			
			if int(fid) in peoples:
				#print("fid is in list")  # DEBUG: SHOW THAT fid in list
				db_cursor.execute( "UPDATE People SET session_time = {} WHERE ID = {};".format(timestayed, int(fid)) )
				 
			else:
				#print("fid is not in list") # DEBUG: SHOW THAT fid not in list
				#db_cursor.execute( "INSERT INTO People(table_id, session_time) VALUES ({}, {});".format(index, timestayed) )
				db_cursor.execute( "INSERT INTO People(session_time) VALUES ({});".format(timestayed) )

			mydb.commit()
			text = 'P{} '.format(fid) + str(timestayed) + 's'
			text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 0.5, 2)[0]
			textX = int(t_x + t_w / 2 - (text_width) / 2)
			textY = int(t_y)
			textLoc = (textX, textY)
			#boxLoc = (textLoc, (textX + text_width - 2, textY - text_height - 2))
			
			# Draw people box, (x, y, z) is the RGB value to change color
			cv2.rectangle(imgDisplay, (t_x, t_y),
						 (t_x + t_w, t_y + t_h),
						  (0, 255, 0), 2)
			
			#cv2.rectangle(imgDisplay, boxLoc[0],
			#			  boxLoc[1],
			#			  (255, 255, 255), cv2.FILLED)

			cv2.putText(imgDisplay, text, textLoc,
						cv2.FONT_HERSHEY_PLAIN,
						1, (0, 77, 0), 1)


if __name__ == "__main__":
	# Read setting for 1 camera only #
	# Later on, add support for multiple camera here with threading if needed #
	model_path = str(cam0['model_path'])
	# input='rtsp://admin:tapway123@tapway1.dahuaddns.com/cam/realmonitor?channel=1&subtype=0 
	# ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true')
	input = str(cam0['input_filename'])
	output = str(cam0['output_filename'])
	frame_interval = int(cam0['frame_interval'])
	threshold = float(cam0['people_threshold'])
	width = int(cam0['output_video_width'])
	height = int(cam0['output_video_height'])

	odapi = od.DetectorAPI(path_to_ckpt=model_path)
	cap = cv2.VideoCapture(input)
	flag, frame = cap.read()
	assert flag == True
	fps = cap.get(cv2.CAP_PROP_FPS)
	
	for i in range(cam0['num_of_tables']):
		tracker[i].videoFrameSize = frame.shape
		tracker[i].fps = fps
	
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	frame_count = 0
	# Define VideoWrite object to output video 
	# cv2.VideoWrite('arg1',arg2,arg3,(width,height))
	# arg1: output file name
	# arg2: Specify Fourcc code
	# arg3: frames per seconds
	# FourCC is a 4-byte code used to specify video codec
	out = cv2.VideoWriter(output, fourcc, fps, (width, height))
	
	### Comment these lines to disable preview
	cv2.namedWindow("Tapway - People Tracking", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("Tapway - People Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	while True:
		r, img = cap.read()

		if r:
			img = cv2.resize(img, (width, height))
			if frame_count % (frame_interval * 5) == 0:
				for i in range(cam0['num_of_tables']): tracker[i].removeDuplicate()

			for i in range(cam0['num_of_tables']): tracker[i].deleteTrack(img)

			if frame_count % frame_interval == 0:
				p1 = (10, 220)
				p2 = (440, 440)
				boxes, scores, classes, num = odapi.processFrame(img[p1[1]:p2[1], p1[0]:p2[0]])
				table_num = 0
				for table in tables: 
					#p1 = (table['x1'], table['y1'])
					#p2 = (table['x2'], table['y2'])
					
					print("Processed table: " + str(table_num+1))
					#roi = img[p3[1]:p4[1], p3[0]:p4[0]]
					#roi = y1 y2 x1 x2
					#roi = img[num['y1']:num['y2'], num['x1']:num['x2']]
					roi = img[p1[1]:p2[1], p1[0]:p2[0]] # slice the image using numpy
					
					# Visualization of the results of a detection.
					for i in range(len(boxes)):
						box = boxes[i]
						x1 = box[1] + p1[0]
						y1 = box[0] + p1[1]
						x2 = box[3] + p1[0]
						y2 = box[2] + p1[1]
						# Class 1 represents human
						if classes[i] == 1 and scores[i] >= threshold:
							matchedID = tracker.getMatchId(img, (x1, y1, x2, y2))
							if matchedID is None:
								id += 1
								tracker.createTrack(img, (x1, y1, x2, y2), str(id), scores[i])
							cv2.rectangle(img,p1,p2,(255,0,0),1)
					table_num = table_num + 1

			#print("the id is " + str(id))  #DEBUG: show id
			drawTrackedPeople(img)
			number = 0
			for n in range(cam0['num_of_tables']):
				number += int(len(tracker.faceTrackers))
			
			for table in tables: 
				p1 = (table['x1'], table['y1'])
				p2 = (table['x2'], table['y2'])
				text = 'Table {} '.format(table['table_id'])
				textX = int(table['x1'])
				textY = int(table['y1'] - 5)
				textLoc = (textX, textY)
				
				# Draw TABLE box, (x, y, z) is the RGB value to change color
				cv2.rectangle(img, p1, p2, (204, 0, 102), 1)
				cv2.putText(img, text, textLoc,
							cv2.FONT_HERSHEY_PLAIN,
							0.75, (0, 0, 0), 1)
						
			cv2.putText(img, 'People: ' + str(number), (0, 25),
						cv2.FONT_HERSHEY_SIMPLEX,
						1, (0, 255, 0), 2)

			out.write(img)
			frame_count += 1

			cv2.imshow("Tapway - People Tracking", img) ### Comment this line to disable preview
			key = cv2.waitKey(1)
			if key & 0xFF == ord('q'):
				break
			cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
		else:
			print('No more frame')
			break
			#raise RuntimeError('No more frame')

			
	cap.release()
	out.release()
	cv2.destroyAllWindows()   