# Import Packages #
import dlib
import numpy as np
import json

# Read JSON Configuration File #
with open('config.json') as f:
    data = json.load(f)
	
# Read from 1 camera: cam0 #
# Later on, add support for multiple camera here with threading if needed #
for camera in data['cameras']:
	if camera['camera_name'] == "cam0": 
		cam0 = camera
		num_of_tables = camera['num_of_tables']
tables = cam0["p"]


class Tracker:

	# Constructor #
	def __init__(self, *args, **kwargs):
		# self.faceTrackers = {'fid'}
		# Object Characteristics
		self.faceTrackers = {}
		self.scores = {}
		self.timeStayed = {}
		self.table = 0
		self.table_x1 = 0
		self.table_y1 = 0
		self.table_x2 = 0
		self.table_y2 = 0
		
		# Object Settings
		self.trackingQuality = 9.5
		self.fidsToDelete = []
		self.fps = None
	
	# Increment dwell time by one (frame) #
	def updateTime(self):
		for fid in self.timeStayed.keys():
			self.timeStayed[fid] = self.timeStayed[fid] + 1

	# Convert dwell time (in frame) to (in second) #
	def getTimeInSecond(self, fid):
		return self.timeStayed[fid] / self.fps

	# Return Table id
	def getTableID(self):
		return self.table
		
	# Return People Num
	def getPeopleNum(self):
		return len(self.faceTrackers)
	
	# Create a new tracker for a new people #
	def createTrack(self, fps, imgDisplay, boundingBox, newFaceID, score, tableID):
		
		# Coordinates of PEOPLE box
		x1, y1, x2, y2 = boundingBox
		
		# Midpoint of x, y
		x_mid = (x1 + x2) / 2
		y_mid = (y1 + y2) / 2
		
		# Create one tracker
		print('Creating new person tracker ' + str(newFaceID))
		tracker = dlib.correlation_tracker()
		tracker.start_track(imgDisplay, dlib.rectangle(x1, y1, x2, y2))
		
		self.faceTrackers[newFaceID] = tracker
		self.scores[newFaceID] = score
		self.timeStayed[newFaceID] = 0
		self.table = int(tableID)
		self.fps = fps
		
		for table in tables:
			if (table['table_id'] == self.table):
				self.table_x1 = table['x1']
				self.table_y1 = table['y1']
				self.table_x2 = table['x2']
				self.table_y2 = table['y2']
		
	# Verify if a people is previously detected #
	def getID(self, imgDisplay, boundingBox, tableID):
	
		# Coordinates of the detected people (CURRENT PEOPLE BOX) #
		x1, y1, x2, y2 = boundingBox
		
		# Midpoint of x, y
		x_mid = (x1 + x2) / 2
		y_mid = (y1 + y2) / 2
		
		# Condition to verify tracked people #
		matchedFid = None
		for fid in self.faceTrackers.keys():
			tracked_position = self.faceTrackers[fid].get_position()
			t_x1 = int(tracked_position.left())
			t_y1 = int(tracked_position.top())
			t_x2 = int(tracked_position.right())
			t_y2 = int(tracked_position.bottom())
			
			t_x_mid = (t_x1 + t_x2) / 2
			t_y_mid = (t_y1 + t_y2) / 2
		
			# if people tracked on same table, continue tracking
			if self.table == int(tableID):
				print("table matched")
				if ((t_x1 <= x_mid <= t_x2) and
				(t_y1 <= y_mid <= t_y2) and
				(x1 <= t_x_mid <= x2) and
				(y1 <= t_y_mid <= y2)):
					self.faceTrackers[fid].start_track(imgDisplay, dlib.rectangle(x1, y1, x2, y2))
					#confidence = self.faceTrackers[fid].update(imgDisplay)
					#matchedFid.append(fid)
					matchedFid = fid
					#return matchedFid
				#else:
				#	self.fidsToDelete.append(fid)
			# else delete this tracker because he has moved out of table
			else:
				self.fidsToDelete.append(fid)
		
		# return matched ID
		return matchedFid
		
	# Delete a tracker if the people is out of bound #
	def deleteTrack(self, imgDisplay):
		
		for fid in self.faceTrackers.keys():
			trackingQuality = self.faceTrackers[fid].update(imgDisplay)
			tracked_position = self.faceTrackers[fid].get_position()
			
			x1 = int(tracked_position.left())
			y1 = int(tracked_position.top())
			x2 = int(tracked_position.right())
			y2 = int(tracked_position.bottom())
			
			x_mid = (x1 + x2) / 2
			y_mid = (y1 + y2) / 2
			
			if not ((self.table_x1 <= x_mid <= self.table_x2) and (self.table_y1 <= y_mid <= self.table_y2)):
				self.fidsToDelete.append(fid)
			
			if trackingQuality < self.trackingQuality:
				self.fidsToDelete.append(fid)
				print("deleting tracker")
				continue
				
		# Delete unwanted tracker
		while len(self.fidsToDelete) > 0:
			fid = self.fidsToDelete.pop()
			self.faceTrackers.pop(fid, None)
			self.scores.pop(fid, None)
			self.timeStayed.pop(fid, None)
			
		# Update time #
		self.updateTime()