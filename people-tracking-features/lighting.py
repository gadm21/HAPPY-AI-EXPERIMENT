# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector
import imutils
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from imutils import contours
from skimage import measure
import time
import track.tracker as track
import track.seattracker as seattrack
import track.cartracker as cartrack
import track.lighttracker as lighttrack
import math


id = 0
carid = 0
frame_interval = 10


def drawTrackedLight(imgDisplay):
    on = 0
    off = 0
    for fid in lighttracker.faceTrackers.keys():
        tracked_position = lighttracker.faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        status = lighttracker.light[fid]
        if status:
            text = "L{} On".format(fid)
            rectColor = (0, 255, 0)
            on += 1
        else:
            text = "L{} Off".format(fid)
            rectColor = (0, 0, 255)
            off += 1

        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        textX = int(t_x + t_w / 2 - (textSize[0]) / 2)
        textY = int(t_y)
        textLoc = (textX, textY - 5)

        cv2.rectangle(imgDisplay, (t_x, t_y), (t_x + t_w, t_y + t_h), rectColor, 2)

        cv2.putText(
            imgDisplay, text, textLoc, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
    return on, off


def detect_bright_spot(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('mask',thresh)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    # cv2.imshow('rain',thresh)
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300 and numPixels < 1000:
            mask = cv2.add(mask, labelMask)
    return mask


lighttracker = lighttrack.Tracker()
x1 = 0
y1 = 100
x2 = 1280
y2 = 360
if __name__ == "__main__":

    cap = cv2.VideoCapture("videos/lighting.mp4")
    flag, frame = cap.read()
    assert flag == True
    height, width, _ = frame.shape
    lighttracker.videoFrameSize = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    frame_count = 0
    total = 0
    per = 1000
    # Define VideoWrite object
    # cv2.VideoWrite('arg1',arg2,arg3,(width,heigh))
    # arg1:output file name
    # arg2:Specify Fourcc code
    # arg3: frames per seconds
    # FourCC is a 4-byte code used to specify video codec
    out = cv2.VideoWriter("output_videos/light.avi", fourcc, fps, (width, height))
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 9000)

    while True:

        r, img = cap.read()
        if r:
            lighttracker.check_status(img)
            lighttracker.deleteTrack(img)
            mask = detect_bright_spot(img)
            cnts = cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            if len(cnts) > 0:
                cnts = contours.sort_contours(cnts)[0]
                for (i, c) in enumerate(cnts):
                    # draw the bright spot on the image
                    (x, y, w, h) = cv2.boundingRect(c)
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h
                    if x1 < x_bar < x2 and y1 < y_bar < y2:
                        matchedID = lighttracker.getMatchId(img, (x, y, x + w, y + h))
                        if matchedID is None:
                            id += 1
                            lighttracker.createTrack(img, (x, y, x + w, y + h), str(id))

            on, off = drawTrackedLight(img)

            cv2.putText(
                img,
                "On: " + str(on),
                (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                "Off: " + str(off),
                (0, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                2,
            )
            out.write(img)
            frame_count += 1
            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
        else:
            break
