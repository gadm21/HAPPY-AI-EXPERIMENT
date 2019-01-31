# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector
import argparse
import sys

import numpy as np
import tensorflow as tf
import cv2
import time
import track.tracker as track
import track.cartracker as cartrack
import math
import scripts.detector as od

# Function to draw out tracked vehicle in image
def drawTrackedVehicle(imgDisplay):
    car = 0
    truck = 0
    motor = 0
    bus = 0
    for fid in cartracker.faceTrackers.keys():
        tracked_position = cartracker.faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())
        t_x_bar = t_x + 0.5 * t_w
        t_y_bar = t_y + 0.5 * t_h
        min_dist = [float('inf'), ]
        StoppedTime = cartracker.getStoppedTime(fid)
        direction = cartracker.direction[fid]
        type = cartracker.type[fid]
        if type == 'Car':
            car += 1
            rectColor = (0, 255, 0)
        elif type == 'Truck':
            truck += 1
            rectColor = (0, 255, 255)
        elif type == 'Motorcycle':
            motor += 1
            rectColor = (255, 255, 0)
        else:
            bus += 1
            rectColor = (255, 0, 0)
        # if StoppedTime>5:
        #     rectColor = (0,0,255)
        #     text = '{}{} Stopped'.format(type,fid) + str(int(StoppedTime)) + 's'
        #
        # else:
        text = '{}{} '.format(type, fid) + str(direction)
        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        textX = int(t_x + t_w / 2 - (textSize[0]) / 2)
        textY = int(t_y)
        textLoc = (textX, textY)

        cv2.rectangle(imgDisplay, (t_x, t_y),
                      (t_x + t_w, t_y + t_h),
                      rectColor, 1)

        cv2.putText(imgDisplay, text, textLoc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    return car, motor, bus, truck

#Function to parse arguments
def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Script for emergency lane driving detection')
    parser.add_argument('-m', '--model',
                        help='Tensorflow object detection model path',
                        required=True,
                        default='model/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb')
    parser.add_argument('-i', '--input',
                        help='Input video filename',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Filename for output video',
                        default='output.avi')
    parser.add_argument('-f', '--frame_interval',
                        help='Amount of frame interval between frame processing',
                        default=5)
    parser.add_argument('-vt', '--vehicle_threshold',
                        help='Threshold value for vehicle detection',
                        default=0.6) # default=0.4 to 0.8 can be used

    results = parser.parse_args(args)
    return (results.model,
            results.input,
            results.output,
            results.frame_interval,
            results.vehicle_threshold)


id = 0
carid = 0
busid = 0
motorid = 0
truckid = 0


if __name__ == "__main__":
    model_path, input, output, frame_interval, vehiclethres = check_arg(sys.argv[1:])
    frame_interval = int(frame_interval)
    vehiclethres = float(vehiclethres)
    odapi = od.DetectorAPI(path_to_ckpt=model_path)
    cap = cv2.VideoCapture(input)
    flag, frame = cap.read()
    assert flag == True
    height, width, _ = frame.shape
    cartracker.videoFrameSize = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    cartracker.fps = fps
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_count = 0
    # Define VideoWrite object
    # cv2.VideoWrite('arg1',arg2,arg3,(width,heigh))
    # arg1:output file name
    # arg2:Specify Fourcc code
    # arg3: frames per seconds
    # FourCC is a 4-byte code used to specify video codec
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 18000)

    #Function to process video
    while True:

        r, img = cap.read()
        if r:
            
            if frame_count % (frame_interval * 3) == 0:
                cartracker.removeDuplicate()
                
            cartracker.deleteTrack(img)
            if frame_count % frame_interval == 0:
                boxes, scores, classes, num = odapi.processFrame(img)
                # Visualization of the results of a detection.
                for i in range(len(boxes)):
                    # Class 3 represents car
                    # cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                    if classes[i] == 3 and scores[i] > vehiclethres:
                        box = boxes[i]
                        matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            carid += 1
                            cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i], 'Car')
                    # Class 4 represents motorcycle
                    elif classes[i] == 4 and scores[i] > vehiclethres - 0.3:
                        box = boxes[i]
                        matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            motorid += 1
                            cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i],
                                                   'Motorcycle')
                    # Class 6 represents Bus
                    elif classes[i] == 6 and scores[i] > vehiclethres:
                        box = boxes[i]
                        matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            busid += 1
                            cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i], 'Bus')
                    # Class 8 represents Truck
                    elif classes[i] == 8 and scores[i] > vehiclethres:
                        box = boxes[i]
                        matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            truckid += 1
                            cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i],
                                                   'Truck')

            # draw all tracked vehicle and return the count of each type
            car, motor, bus, truck = drawTrackedVehicle(img)
            # placing the vehicle count on top left corner
            cv2.putText(img, 'Cars: ' + str(carid), (50, 63),#cv2.putText(img, 'Cars: ' + str(car), (50, 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2) # font_scale 0.5
            #cv2.putText(img, 'Motor: ' + str(motor), (50, 33),
            cv2.putText(img, 'Motor: ' + str(motorid), (50, 100),# (50, 83) 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2)
            cv2.putText(img, 'Bus: ' + str(busid), (50, 137),# (50, 103)
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 128, 0), 2)
            cv2.putText(img, 'Truck: ' + str(truckid), (50, 174),# (50, 123)52+21
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

            out.write(img)
            frame_count += 1
            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        else:
            raise RuntimeError('No more frame')
            
