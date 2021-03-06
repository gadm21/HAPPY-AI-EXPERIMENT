# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector
import argparse
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

import track.cartracker as cartrack
import track.tracker as track
import detector as od


tracker = track.Tracker()
cartracker = cartrack.Tracker()
cartracker.trackingQuality = 7
cartracker.outOfScreenThreshold = 0.2


def drawTrackedFace(imgDisplay):
    for fid in tracker.faceTrackers.keys():
        tracked_position = tracker.faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())
        direction = tracker.direction[fid]
        text = 'P{} '.format(fid) + str(direction)

        t_x_bar = t_x + 0.5 * t_w
        t_y_bar = t_y + 0.5 * t_h

        p1 = np.array([pt1[0], pt1[1]])
        p2 = np.array([pt2[0], pt2[1]])
        p3 = np.array([t_x_bar, t_y_bar])

        d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

        if abs(d) < 50 and not counted[fid]:
            if direction == 'up':
                global up
                up += 1
                counted[fid] = True

            elif direction == 'down':
                global down
                down += 1
                counted[fid] = True

        if direction == 'up':
            rectColor = (0, 0, 255)
        elif direction == 'down':
            rectColor = (255, 255, 0)
        else:
            rectColor = (0, 255, 0)

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


def drawTrackedVehicle(imgDisplay):
    for fid in cartracker.faceTrackers.keys():
        tracked_position = cartracker.faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())
        t_x_bar = t_x + 0.5 * t_w
        t_y_bar = t_y + 0.5 * t_h
        score = cartracker.scores[fid]
        direction = cartracker.direction[fid]
        type = cartracker.type[fid]

        p1 = np.array([pt1[0], pt1[1]])
        p2 = np.array([pt2[0], pt2[1]])
        p3 = np.array([t_x_bar, t_y_bar])

        d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        print(d)
        if abs(d) < 100 and not carcounted[fid]:
            if direction == 'up':
                global up
                up += 1
                carcounted[fid] = True

            elif direction == 'down':
                global down
                down += 1
                carcounted[fid] = True

        if direction == 'up':
            rectColor = (0, 0, 255)

        elif direction == 'down':
            rectColor = (255, 255, 0)

        else:
            rectColor = (0, 255, 0)
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


def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Script for intrusion detection')
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
    parser.add_argument('-p', '--points',
                        help='Points of the line for intrusion detection in the format of (x1,y1,x2,y2)',
                        required=True,
                        default='(100,200,600,200)')
    parser.add_argument('-vt', '--vehicle_threshold',
                        help='Threshold value for vehicle detection',
                        default=0.8)
    parser.add_argument('-pt', '--people_threshold',
                        help='Threshold value for people detection',
                        default=0.8)

    results = parser.parse_args(args)
    return (results.model,
            results.input,
            results.output,
            results.frame_interval,
            results.points,
            results.vehicle_threshold,
            results.people_threshold)


id = 0
carid = 0

pt1 = None
pt2 = None
up = 0
down = 0
counted = {}
carcounted = {}

if __name__ == "__main__":
    model_path, input, output, frame_interval, points, vehiclethres, threshold = check_arg(sys.argv[1:])
    frame_interval = int(frame_interval)
    vehiclethres = float(vehiclethres)
    threshold = float(threshold)
    points = eval(points)
    pt1 = (points[0], points[1])
    pt2 = (points[2], points[3])
    odapi = od.DetectorAPI(path_to_ckpt=model_path)
    cap = cv2.VideoCapture(input)
    # cap = cv2.VideoCapture('videos/Mainline-Detect Vehicle type/A0014.mov')
    flag, frame = cap.read()
    assert flag == True
    height, width, _ = frame.shape
    tracker.videoFrameSize = frame.shape
    cartracker.videoFrameSize = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker.fps = fps
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

    while True:
        r, img = cap.read()
        if r:
            tracker.deleteTrack(img)
            cartracker.deleteTrack(img)

            boxes, scores, classes, num = odapi.processFrame(img)
            # Visualization of the results of a detection.
            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] >= threshold:
                    box = boxes[i]
                    matchedID = tracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                    if matchedID is None:
                        id += 1
                        tracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(id), scores[i])
                        counted[str(id)] = False
                elif classes[i] == 3 and scores[i] > vehiclethres:
                    box = boxes[i]
                    matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                    if matchedID is None:
                        carid += 1
                        cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i], 'Car')
                        carcounted[str(carid)] = False
                elif classes[i] == 4 and scores[i] > vehiclethres - 0.3:
                    box = boxes[i]
                    matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                    if matchedID is None:
                        carid += 1
                        cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i],
                                               'Motorcycle')
                        carcounted[str(carid)] = False

                elif classes[i] == 6 and scores[i] > vehiclethres:
                    box = boxes[i]
                    matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                    if matchedID is None:
                        carid += 1
                        cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i], 'Bus')
                        carcounted[str(carid)] = False

                elif classes[i] == 8 and scores[i] > vehiclethres:
                    box = boxes[i]
                    matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                    if matchedID is None:
                        carid += 1
                        cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i], 'Truck')
                        carcounted[str(carid)] = False

                # cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)

            drawTrackedFace(img)
            drawTrackedVehicle(img)
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)

            cv2.putText(img, 'Up: ' + str(up), (0, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv2.putText(img, 'Down: ' + str(down), (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2)

            out.write(img)
            frame_count += 1
            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        else:
            raise RuntimeError('No more frame')
