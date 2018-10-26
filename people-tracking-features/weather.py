# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from skimage import measure
import time
import track.tracker as track
import track.seattracker as seattrack
import track.cartracker as cartrack
import math


id = 0
carid = 0
frame_interval = 10

def detect_bright_spot(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('mask',thresh)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=3)
    # cv2.imshow('rain',thresh)
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    total = 0
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
        if numPixels > 1200:
            total+=numPixels
            mask = cv2.add(mask, labelMask)
    # cv2.imshow('measured',mask)

    return total

if __name__ == "__main__":
    cap = cv2.VideoCapture('videos/Mainline-Raining/A0069.mov')
    flag, frame = cap.read()
    assert flag == True
    height, width, _ = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_count = 0
    total=0
    per=1000
    # Define VideoWrite object
    # cv2.VideoWrite('arg1',arg2,arg3,(width,heigh))
    # arg1:output file name
    # arg2:Specify Fourcc code
    # arg3: frames per seconds
    # FourCC is a 4-byte code used to specify video codec
    out = cv2.VideoWriter('output_videos/weather.avi', fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 9000)

    while True:

        r, img = cap.read()
        if r:
            total = detect_bright_spot(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            intensity_variance_per_row = np.var(gray, axis=0)
            avg_variance = np.average(intensity_variance_per_row, axis=0)
            # if(frame_count % frame_interval) == 0:
            #     per = foggy(img)
            if total > 20000:
                text = 'Raining'
                color = (0,0,255)
            else:
                text = 'Not Raining'
                color = (255,255,255)

            cv2.putText(img, str(total), (0, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            cv2.putText(img, str(text), (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)
            if avg_variance < 3900:
                text = 'Foggy'
                color = (0,0,255)
            else:
                text = 'Not Foggy'
                color = (255,255,255)
            cv2.putText(img, str(int(avg_variance)), (0, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255,255,255), 2)
            cv2.putText(img, str(text), (0, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)

            out.write(img)
            frame_count+=1
            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        else:
            break

