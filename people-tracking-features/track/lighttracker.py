import dlib
import numpy as np
import cv2


class Tracker:
    def __init__(self, *args, **kwargs):
        self.faceTrackers = {}
        self.faceID = {}
        self.fidsToDelete = []
        self.trackingQuality = 6
        self.videoFrameSize = np.empty((0, 0, 0))
        self.outOfScreenThreshold = 0.5
        self.light = {}

    def createTrack(self, imgDisplay, boundingBox, currentFaceID):

        x1, y1, x2, y2 = boundingBox

        w = int(x2 - x1)
        h = int(y2 - y1)
        x = int(x1)
        y = int(y1)

        print("Creating new seat tracker" + str(currentFaceID))
        tracker = dlib.correlation_tracker()
        tracker.start_track(
            imgDisplay, dlib.rectangle(x - 20, y - 20, x + w + 20, y + h + 20)
        )
        # tracker.start_track(imgDisplay, dlib.rectangle(x, y, x + w, y + h))
        # tracker.start_track(imgDisplay,dlib.rectangle(x-50,y-50,x+w+50,y+h+50))

        self.faceTrackers[currentFaceID] = tracker
        self.light[currentFaceID] = True

    def deleteAll(self):
        for fid in self.faceTrackers.keys():
            self.fidsToDelete.append(fid)

    def appendDeleteFid(self, fid):
        self.fidsToDelete.append(fid)

    def getRelativeOutScreen(self, fid):
        tracked_position = self.faceTrackers[fid]

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

    def check_status(self, imgDisplay):
        img = imgDisplay.copy()
        for fid in self.faceTrackers.keys():
            tracked_position = self.faceTrackers[fid].get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            light_img = img[t_y : t_y + t_h, t_x : t_x + t_w]
            gray = cv2.cvtColor(light_img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
            numPixels = cv2.countNonZero(thresh)
            if numPixels > 100:
                self.light[fid] = True
            else:
                self.light[fid] = False

    def deleteTrack(self, imgDisplay):
        for fid in self.faceTrackers.keys():
            if self.light[fid]:
                trackingQuality = self.faceTrackers[fid].update(imgDisplay)

                if trackingQuality < self.trackingQuality:
                    self.fidsToDelete.append(fid)
                    continue

            relativeOutScreen = self.getRelativeOutScreen(fid)

            if relativeOutScreen > self.outOfScreenThreshold:
                print("Face Out of Screen")
                self.fidsToDelete.append(fid)

        while len(self.fidsToDelete) > 0:
            fid = self.fidsToDelete.pop()
            self.faceTrackers.pop(fid, None)
            self.light.pop(fid, None)

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
        for fid in self.faceTrackers.keys():
            tracked_position = self.faceTrackers[fid].get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())

            t_x_bar = t_x + 0.5 * t_w
            t_y_bar = t_y + 0.5 * t_h

            if (
                (t_x <= x_bar <= (t_x + t_w))
                and (t_y <= y_bar <= (t_y + t_h))
                or (x <= t_x_bar <= (x + w))
                and (y <= t_y_bar <= (y + h))
            ):
                matchedFid = fid

                self.faceTrackers[fid].start_track(
                    imgDisplay, dlib.rectangle(x - 20, y - 20, x + w + 20, y + h + 20)
                )
                # self.faceTrackers[fid].start_track(imgDisplay, dlib.rectangle(x, y, x + w, y + h))
                # self.faceTrackers[fid].start_track(imgDisplay,dlib.rectangle(x-50,y-50,x+w+50,y+h+50))

        return matchedFid

    def removeDuplicate(self):
        for id in self.faceTrackers.copy().keys():
            tracked_position = self.faceTrackers[id].get_position()
            x = int(tracked_position.left())
            y = int(tracked_position.top())
            w = int(tracked_position.width())
            h = int(tracked_position.height())
            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            for fid in self.faceTrackers.copy().keys():
                if id == fid:
                    continue
                tracked_position = self.faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                if (
                    (t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h))
                ) and ((x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                    self.faceTrackers.pop(id, None)
                    print("delete overlap {}, {}".format(id, fid))
