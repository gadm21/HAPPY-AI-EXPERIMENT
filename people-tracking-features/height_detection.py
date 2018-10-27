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


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        # print("Elapsed Time:", end_time-start_time)

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),
                             int(boxes[0, i, 1] * im_width),
                             int(boxes[0, i, 2] * im_height),
                             int(boxes[0, i, 3] * im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Script for overheight vehicle detection')
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
                        help='Points of the line for overheight detection in the format of (x1,y1,x2,y2)',
                        required=True,
                        default='(100,200,600,200)')

    parser.add_argument('-vt', '--vehicle_threshold',
                        help='Threshold value for vehicle detection',
                        default=0.85)

    results = parser.parse_args(args)
    return (results.model,
            results.input,
            results.output,
            results.frame_interval,
            results.points,
            results.vehicle_threshold)


tracker = track.Tracker()
cartracker = cartrack.Tracker()


def drawTrackedVehicle(imgDisplay):
    for fid in cartracker.faceTrackers.keys():
        tracked_position = cartracker.faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())
        t_x_bar = t_x + 0.5 * t_w
        t_y_bar = t_y + 0.5 * t_h

        p1 = np.array([pt1[0], pt1[1]])
        p2 = np.array([pt2[0], pt2[1]])
        p3 = np.array([t_x, t_y])

        d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        if abs(d) < 75:
            if (t_h) > 240:
                overheight[fid] = 2
            elif (t_h) > 200:
                overheight[fid] = 1
            else:
                overheight[fid] = 0
        else:
            overheight[fid] = 0
        type = cartracker.type[fid]
        rectColor = (0, 255, 0)
        text = '{}{}'.format(type, fid)
        if overheight[fid] == 2:
            text += ' Overheight'
            rectColor = (0, 0, 255)
            print('Overheight vehicle reached line {}'.format(fid))
        elif overheight[fid] == 1:
            text += ' Warning'
            print('Warning, overheight vehicle approaching line {}'.format(fid))
            rectColor = (255, 0, 0)
        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        textX = int(t_x + t_w / 2 - (textSize[0]) / 2)
        textY = int(t_y)
        textLoc = (textX, textY)

        cv2.rectangle(imgDisplay, (t_x, t_y),
                      (t_x + t_w, t_y + t_h),
                      rectColor, 1)

        cv2.putText(imgDisplay, text, textLoc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)


def drawTrackedFace(imgDisplay):
    for fid in tracker.faceTrackers.keys():
        tracked_position = tracker.faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        confidence = tracker.scores[fid]
        text = 'P{} '.format(fid) + str(confidence)
        rectColor = (0, 255, 0)

        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        textX = int(t_x + t_w / 2 - (textSize[0]) / 2)
        textY = int(t_y)
        textLoc = (textX, textY)

        cv2.rectangle(imgDisplay, (t_x, t_y),
                      (t_x + t_w, t_y + t_h),
                      rectColor, 1)

        cv2.putText(imgDisplay, text, textLoc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)


id = 0
carid = 0
overheight = {}
pt1 = None
pt2 = None

if __name__ == "__main__":
    model_path, input, output, frame_interval, points, vehiclethres = check_arg(sys.argv[1:])
    points = eval(points)
    pt1 = (points[0], points[1])
    pt2 = (points[2], points[3])
    odapi = DetectorAPI(path_to_ckpt=model_path)
    cap = cv2.VideoCapture(input)
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)

    while True:

        r, img = cap.read()
        if r:
            tracker.deleteTrack(img)
            cartracker.deleteTrack(img)
            if frame_count % 1 == 0:
                boxes, scores, classes, num = odapi.processFrame(img)
                # Visualization of the results of a detection.
                for i in range(len(boxes)):
                    # Class 1 represents human
                    # cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                    if classes[i] == 3 and scores[i] > vehiclethres:
                        box = boxes[i]
                        matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            carid += 1
                            cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i], 'Car')
                            overheight[str(carid)] = 0
                    elif classes[i] == 4 and scores[i] > vehiclethres - 0.3:
                        box = boxes[i]
                        matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            carid += 1
                            cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i],
                                                   'Motorcycle')
                            overheight[str(carid)] = 0

                    elif classes[i] == 6 and scores[i] > vehiclethres:
                        box = boxes[i]
                        matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            carid += 1
                            cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i], 'Bus')
                            overheight[str(carid)] = 0

                    elif classes[i] == 8 and scores[i] > vehiclethres:
                        box = boxes[i]
                        matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            carid += 1
                            cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i],
                                                   'Truck')
                            overheight[str(carid)] = 0

            drawTrackedVehicle(img)
            cv2.line(img, pt1, pt2, (0, 0, 255), 2)

            number = int(len(cartracker.faceTrackers))
            cv2.putText(img, 'Cars: ' + str(number), (0, 25),
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
