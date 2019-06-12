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
		self.table = {}
		
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
	def getTableID(self, fid):
		return self.table[fid]
	
	# Create a new tracker for a new people #
	def createTrack(self, imgDisplay, boundingBox, newFaceID, tableID, score):
		
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
		self.table[newFaceID] = int(tableID)
		
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
			# if people tracked on same table, continue tracking
			if self.table[fid] == int(tableID):
				self.faceTrackers[fid].start_track(imgDisplay, dlib.rectangle(x1, y1, x2, y2))
				#self.faceTrackers[fid].update(imgDisplay)
				matchedFid = fid
			# else delete this tracker because he has moved out of table
			#elif self.table[fid] in range(1, num_of_tables + 1):
			#	self.fidsToDelete.append(fid)
			#else:
			#	pass
		
		# return matched ID
		return matchedFid
		
	# Delete a tracker if the people is out of bound #
	def deleteTrack(self, imgDisplay):
		
		for fid in self.faceTrackers.keys():
			trackingQuality = self.faceTrackers[fid].update(imgDisplay)
			
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
			self.table.pop(fid, None)
			
		# Update time #
		self.updateTime()