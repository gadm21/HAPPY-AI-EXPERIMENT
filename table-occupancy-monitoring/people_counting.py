# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

# Import Libraries #
import os
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
tracker = {}
for i in range(1, cam0['num_of_tables']+1):
	tracker[i] = track.Tracker()

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
	#db_cursor.execute( "SELECT ID FROM people" )
	#peoples = [item['ID'] for item in db_cursor.fetchall()]
	for index in tracker.keys():
		for fid in tracker[index].faceTrackers.keys():
			db_cursor.execute( "SELECT ID FROM people WHERE ID = {}".format(fid) )
			this_id = db_cursor.fetchone()

			#print(fid) # DEBUG: display current person id in processing
			tracked_position = tracker[index].faceTrackers[fid].get_position()
			t_x = int(tracked_position.left())
			t_y = int(tracked_position.top())
			t_w = int(tracked_position.width())
			t_h = int(tracked_position.height())

			timestayed = int(tracker[index].getTimeInSecond(fid))
			table = int(tracker[index].getTableID())
			
			if this_id is not None: #int(fid) in peoples:
				#print("fid is in list")  # DEBUG: SHOW THAT fid in list
				db_cursor.execute( "UPDATE people SET dwell_time = {} WHERE ID = {};".format(timestayed, fid) )
				 
			else:
				#print("fid is not in list") # DEBUG: SHOW THAT fid not in list
				db_cursor.execute( "INSERT INTO people(table_id, dwell_time) VALUES ({}, {});".format(table, timestayed) )
				#db_cursor.execute( "INSERT INTO people(session_time) VALUES ({});".format(timestayed) )
			
			text = 'P{} '.format(fid) + str(timestayed) + 's'

			textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 0.5, 2)[0]
			textX = int(t_x + t_w / 2 - (textSize[0]) / 2)
			textY = int(t_y)
			textLoc = (textX, textY)
			
			# Draw people box, (x, y, z) is the RGB value to change color
			cv2.rectangle(imgDisplay, (t_x, t_y),
						 (t_x + t_w, t_y + t_h),
						  (0, 255, 0), 2)

			cv2.putText(imgDisplay, text, textLoc,
						cv2.FONT_HERSHEY_PLAIN,
						1.5, (0, 77, 0), 2)


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
	matchedID = []
	
	while True:
		r, img = cap.read()
		
		if r:
			img = cv2.resize(img, (width, height))
			
			for index in tracker.keys():
				tracker[index].deleteTrack(img)

			if frame_count % frame_interval == 0:
				# reduce zone size to increase accuracy 
				# ##please take note, this is hardcoded, please modify## #
				p1 = (10, 200)
				p2 = (1342, 1066)
				boxes, scores, classes, num = odapi.processFrame(img[p1[1]:p2[1], p1[0]:p2[0]])
				## cv2.rectangle(img,p1,p2,(255,0,0),2)
				table_num = 0
				tableID = -1
				
				# Visualization of the results of a detection.
				for i in range(len(boxes)):
					box = boxes[i]
					x1 = box[1] + p1[0]
					y1 = box[0] + p1[1]
					x2 = box[3] + p1[0]
					y2 = box[2] + p1[1]
					x_mid = (x1 + x2) / 2
					y_mid = (y1 + y2) / 2
					# Class 1 represents human
					if classes[i] == 1 and scores[i] >= threshold:
						#print("detected people")
						for table in tables: 
							if((table['x1'] <= x_mid <= table['x2']) and (table['y1'] <= y_mid <= table['y2'])):
								tableID = table['table_id']
								break
							else:
								tableID = -1
						
						# If PEOPLE within zone
						if tableID != -1:
							matchedID = tracker[tableID].getID(img, (x1, y1, x2, y2), tableID)
							#print("matched id is " + str (matchedID))
							if matchedID is None:
								id += 1
								tracker[tableID].createTrack(fps, img, (x1, y1, x2, y2), str(id), scores[i], tableID)
								
					table_num = table_num + 1

			#print("the id is " + str(id))  #DEBUG: show id
			drawTrackedPeople(img)
			
			# Draw TABLE box, (x, y, z) is the RGB value to change color
			location = 45
			for table in tables: 
				p1 = (table['x1'], table['y1'])
				p2 = (table['x2'], table['y2'])
				text = 'Table {} '.format(table['table_id'])
				textX = int(table['x1'])
				textY = int(table['y1'] - 5)
				textLoc = (textX, textY)
				
				cv2.rectangle(img, p1, p2, (204, 0, 102), 2)
				cv2.putText(img, text, textLoc,
							cv2.FONT_HERSHEY_PLAIN,
							2, (0, 0, 0), 2)
			
				# Show PEOPLE total in the frame
				cv2.putText(img, 'Table {}: '.format(table['table_id']) + str(tracker[table['table_id']].getPeopleNum()), (10, location),
							cv2.FONT_HERSHEY_SIMPLEX,
							1, (0, 255, 0), 2)
				location += 40

			out.write(img)
			frame_count += 1

			cv2.imshow("Tapway - People Tracking", img) ### Comment this line to disable preview
			key = cv2.waitKey(1)
			if key & 0xFF == ord('q'):
				mydb.commit()
				break
			cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
			
		else:
			print('No more frame')
			mydb.commit()
			#break
			#raise RuntimeError('No more frame')

			
	cap.release()
	out.release()
	cv2.destroyAllWindows()   
