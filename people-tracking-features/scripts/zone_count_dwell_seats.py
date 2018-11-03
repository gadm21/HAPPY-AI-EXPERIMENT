# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector
import argparse
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

import track.seattracker as seattrack
import track.tracker as track
import detector as od


tracker = track.Tracker()
seattracker = seattrack.Tracker()


def drawTrackedFace(imgDisplay):
    for fid in tracker.faceTrackers.keys():
        tracked_position = tracker.faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        timestayed = int(tracker.getTimeInSecond(fid))
        text = 'P{} '.format(fid) + str(timestayed) + 's'
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
                    0.5, (0, 0, 0), 1)


def drawTrackedSeat(imgDisplay):
    for fid in seattracker.faceTrackers.keys():
        tracked_position = seattracker.faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        text = 'S{}'.format(fid)
        occupiedbool = seattracker.occupied[fid]
        if occupiedbool:
            rectColor = (255, 0, 0)
        else:
            rectColor = (0, 0, 255)

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
    parser = argparse.ArgumentParser(description='Script for counting person in zone and free/occupied seat detection')
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
    parser.add_argument('-p', '--roi_points',
                        help='Points for ROI in format of (x1,y1,x2,y2)',
                        required=True)
    parser.add_argument('-pt', '--people_threshold',
                        help='Threshold value for people detection',
                        default=0.9)
    parser.add_argument('-st', '--seat_threshold',
                        help='Threshold value for seat detection',
                        default=0.9)
    results = parser.parse_args(args)
    return (results.model,
            results.input,
            results.output,
            results.frame_interval,
            results.roi_points,
            results.seat_threshold,
            results.people_threshold)


id = 0
seat = 0
p1 = None
p2 = None
if __name__ == "__main__":
    model_path, input, output, frame_interval, points, seatthreshold, threshold = check_arg(sys.argv[1:])
    frame_interval = int(frame_interval)
    seatthreshold = float(seatthreshold)
    threshold = float(threshold)
    points = eval(points)
    p1 = (points[0], points[1])
    p2 = (points[2], points[3])
    odapi = od.DetectorAPI(path_to_ckpt=model_path)
    cap = cv2.VideoCapture(input)
    flag, frame = cap.read()
    assert flag == True
    tracker.videoFrameSize = frame.shape
    seattracker.videoFrameSize = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker.fps = fps
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_count = 0
    # Define VideoWrite object
    # cv2.VideoWrite('arg1',arg2,arg3,(width,heigh))
    # arg1:output file name
    # arg2:Specify Fourcc code
    # arg3: frames per seconds
    # FourCC is a 4-byte code used to specify video codec
    out = cv2.VideoWriter(output, fourcc, 25.0, (768, 432))

    while True:
        r, img = cap.read()
        if r:
            if frame_count % (frame_interval * 5) == 0:
                tracker.removeDuplicate()
                # seattracker.removeDuplicate()

            tracker.deleteTrack(img)
            seattracker.deleteTrack(img)

            seattracker.checkOccupied(tracker)

            if frame_count % frame_interval == 0:
                roi = img[p1[1]:p2[1], p1[0]:p2[0]]
                boxes, scores, classes, num = odapi.processFrame(roi)
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
                        # cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                    elif (classes[i] == 2) and scores[i] >= seatthreshold:
                        matchedID = seattracker.getMatchId(img, (x1, y1, x2, y2))
                        if matchedID is None:
                            seat += 1
                            seattracker.createTrack(img, (x1, y1, x2, y2), str(seat), scores[i])

            drawTrackedSeat(img)
            drawTrackedFace(img)
            number = int(len(tracker.faceTrackers))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            cv2.putText(img, 'People: ' + str(number), (0, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            occupied, unoccupied = seattracker.getAmount()
            cv2.putText(img, 'Occupied Seat: {}'.format(str(occupied)), (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)
            cv2.putText(img, 'Free Seat: {}'.format(str(unoccupied)), (0, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            out.write(img)
            frame_count += 1
            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        else:
            raise RuntimeError('No more frame')
