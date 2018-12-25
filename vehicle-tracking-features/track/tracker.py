import dlib
import numpy as np
import vehicle as veh
import math


class Tracker:
    def __init__(self, *args, **kwargs):
        self.objectTrackers = {}
        self.objectID = {}
        self.fidsToDelete = []
        self.trackingQuality = 10
        self.videoFrameSize = np.empty((0, 0, 0))
        self.outOfScreenThreshold = 0.4
        self.vehicleList = {}
        self.fps = None
        # self.speedLimit = 50
        self.speedLimit = 1000

    def createTrack(self, imgDisplay, boundingBox, currentobjectID):

        x1, y1, x2, y2 = boundingBox

        w = int(x2 - x1)
        h = int(y2 - y1)
        x = int(x1)
        y = int(y1)

        # print("Creating new tracker" + str(currentobjectID))
        tracker = dlib.correlation_tracker()
        tracker.start_track(imgDisplay, dlib.rectangle(x, y, x + w, y + h))

        self.objectTrackers[currentobjectID] = tracker

    def deleteAll(self):
        for fid in self.objectTrackers.keys():
            self.fidsToDelete.append(fid)

    def appendDeleteFid(self, fid):
        self.fidsToDelete.append(fid)

    def getRelativeOutScreen(self, fid):
        tracked_position = self.objectTrackers[fid]

        left = tracked_position.get_position().left()
        right = tracked_position.get_position().right()
        top = tracked_position.get_position().top()
        bottom = tracked_position.get_position().bottom()

        width = tracked_position.get_position().width()
        height = tracked_position.get_position().height()

        area = width * height

        outScreenX = 0
        outScreenY = 0

        if left < 0:
            outScreenX += 0 - left

        if right > self.videoFrameSize[1]:
            outScreenX += right - self.videoFrameSize[1]

        if top < 0:
            outScreenY += 0 - top

        if bottom > self.videoFrameSize[0]:
            outScreenY += bottom - self.videoFrameSize[0]

        outScreenIntersect = outScreenX * outScreenY
        outScreenArea = outScreenX * height + outScreenY * width - outScreenIntersect
        relativeOutScreen = outScreenArea / area

        return relativeOutScreen

    def deleteTrack(self, imgDisplay):
        for fid in self.objectTrackers.keys():

            trackingQuality = self.objectTrackers[fid].update(imgDisplay)

            if trackingQuality < self.trackingQuality:
                self.fidsToDelete.append(fid)
                continue

            tracked_position = self.objectTrackers[fid]

            width = tracked_position.get_position().width()
            height = tracked_position.get_position().height()

            if width < 20.0 or height < 20.0:
                self.fidsToDelete.append(fid)
                continue
                # print(width,height,'height')

            relativeOutScreen = self.getRelativeOutScreen(fid)

            if relativeOutScreen > self.outOfScreenThreshold:
                # print("object Out of Screen")
                self.fidsToDelete.append(fid)

            left = tracked_position.get_position().left()
            top = tracked_position.get_position().top()

            centerX = left + (width / 2.0)
            centerY = top + (height / 2.0)

            preX = self.vehicleList[fid].centerX
            preY = self.vehicleList[fid].centerY

            # deltaY = abs((720-centerY)*(720-centerY)*(720-centerY)/2.0-(720-preY)*(720-preY)*(720-preY)/2.0) 
            deltaY = abs((720-centerY)*(720-centerY)/2.0-(720-preY)*(720-preY)/2.0) 

            # deltaX = centerX - preX
            # deltaY = centerY - preY
            pixelDist = deltaY/500
            # pixelDist = math.sqrt(deltaX * deltaX + deltaY * deltaY)
            if self.vehicleList[fid].speed==0:
                self.vehicleList[fid].speed = pixelDist*self.fps
            else:
                self.vehicleList[fid].speed = (self.vehicleList[fid].speed*4+pixelDist*self.fps)/5

            # self.vehicleList[fid].speed = pixelDist * self.fps

            self.vehicleList[fid].centerX = centerX
            self.vehicleList[fid].centerY = centerY

            if self.vehicleList[fid].speed < self.speedLimit:
                self.vehicleList[fid].stay += 1
            else:
                self.vehicleList[fid].stay = 0

        while len(self.fidsToDelete) > 0:
            fid = self.fidsToDelete.pop()
            self.objectTrackers.pop(fid, None)

    def getMatchId(self, imgDisplay, boundingBox):

        x1, y1, x2, y2 = boundingBox

        w = int(x2 - x1)
        h = int(y2 - y1)
        x = int(x1)
        y = int(y1)

        ##calculate centerpoint
        x_bar = x + 0.5 * w
        y_bar = y + 0.5 * h

        matchedFid = None
        for fid in self.objectTrackers.keys():
            tracked_position = self.objectTrackers[fid].get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())

            t_x_bar = t_x + 0.5 * t_w
            t_y_bar = t_y + 0.5 * t_h

            if (
                (t_x <= x_bar <= (t_x + t_w))
                and (t_y <= y_bar <= (t_y + t_h))
                and (x <= t_x_bar <= (x + w))
                and (y <= t_y_bar <= (y + h))
            ):
                matchedFid = fid

                self.objectTrackers[fid].start_track(
                    imgDisplay, dlib.rectangle(x, y, x + w, y + h)
                )

        return matchedFid
