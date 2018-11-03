# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector
import argparse
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

import track.bagtracker as bagtrack
import track.tracker as track
import detector as od


tracker = track.Tracker()
tracker.trackingQuality = 9
bagtracker = bagtrack.Tracker()


def drawTrackedBag(imgDisplay):
    for fid in bagtracker.faceTrackers.keys():
        tracked_position = bagtracker.faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        owner = bagtracker.owner[fid]
        abandoned = bagtracker.abandoned[fid]
        if owner is not None:
            owner = 'Owner: P' + str(owner)
        else:
            owner = 'No Owner'
        if abandoned:
            rectColor = (0, 0, 255)
            owner = owner + '(Abandoned)'
            print('detected abandoned package {} owner: {}'.format(fid, bagtracker.owner[fid]))
        else:
            rectColor = (255, 0, 0)

        text = '{}'.format(owner)

        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        textX = int(t_x + t_w / 2 - (textSize[0]) / 2)
        textY = int(t_y)
        textLoc = (textX, textY)

        cv2.rectangle(imgDisplay, (t_x, t_y),
                      (t_x + t_w, t_y + t_h),
                      rectColor, 1)

        cv2.putText(imgDisplay, text, textLoc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)


def drawTrackedFace(imgDisplay):
    for fid in tracker.faceTrackers.keys():
        tracked_position = tracker.faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        text = 'P{}'.format(fid)
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
                    0.5, (255, 255, 255), 2)


def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Script for unattended package detection')
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
    parser.add_argument('-bt', '--bag_threshold',
                        help='Threshold value for bag detection',
                        default=0.5)
    parser.add_argument('-pt', '--person_threshold',
                        help='Threshold value for person detection',
                        default=0.5)

    results = parser.parse_args(args)
    return (results.model,
            results.input,
            results.output,
            results.frame_interval,
            results.bag_threshold,
            results.person_threshold)


id = 0
bag = 0
if __name__ == "__main__":
    model_path, input, output, frame_interval, bagthreshold, threshold = check_arg(sys.argv[1:])
    bagthreshold = float(bagthreshold)
    threshold = float(threshold)
    frame_interval = int(frame_interval)
    odapi = od.DetectorAPI(path_to_ckpt=model_path)
    cap = cv2.VideoCapture(input)
    flag, frame = cap.read()
    assert flag == True
    tracker.videoFrameSize = frame.shape
    bagtracker.videoFrameSize = frame.shape
    height, width, _ = frame.shape
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
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    while True:
        r, img = cap.read()
        if r:
            if frame_count % (frame_interval * 5) == 0:
                tracker.removeDuplicate()
                # seattracker.removeDuplicate()

            tracker.deleteTrack(img)
            bagtracker.deleteTrack(img)
            bagtracker.checkOwner(tracker)
            if frame_count % frame_interval == 0:
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
                        # cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                    elif (classes[i] == 27 or classes[i] == 31 or classes[i] == 33) and scores[i] > bagthreshold:
                        box = boxes[i]
                        matchedID = bagtracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            bag += 1
                            bagtracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(bag), scores[i])

            drawTrackedFace(img)
            drawTrackedBag(img)

            out.write(img)
            frame_count += 1
            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        else:
            raise RuntimeError('No more frame')
