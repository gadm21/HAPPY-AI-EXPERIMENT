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
import track.seattracker as seattrack
import track.cartracker as cartrack
import math
import detector as od

class HoughBundler:
    '''Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        return math.degrees(orientation)

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        '''Check if line have enough distance and angle to be count as similar
        '''
        for group in groups:
            # walk through existing line groups
            for line_old in group:
                # check distance
                if self.get_distance(line_old, line_new) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    # if all is ok -- line is similar to others in group
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        # if it is totally different line
        return True

    def DistancePointLine(self, point, line):
        """Get distance between point and line
        http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        """
        px, py = point
        x1, y1, x2, y2 = line

        def lineMagnitude(x1, y1, x2, y2):
            'Get line (aka vector) length'
            lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return lineMagnitude

        LineMag = lineMagnitude(x1, y1, x2, y2)
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = lineMagnitude(px, py, x1, y1)
            iy = lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """
        dist1 = self.DistancePointLine(a_line[:2], b_line)
        dist2 = self.DistancePointLine(a_line[2:], b_line)
        dist3 = self.DistancePointLine(b_line[:2], a_line)
        dist4 = self.DistancePointLine(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # Parameters to play with
        min_distance_to_merge = 50
        min_angle_to_merge = 20
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if (len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        if 45 < orientation < 135:
            # sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            # sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv2.HoughLinesP()
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation < 135:
                lines_y.append(line_i)
            else:
                lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_x, lines_y]:
            if len(i) > 0:
                groups = self.merge_lines_pipeline_2(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_lines_segments1(group))

                merged_lines_all.extend(merged_lines)

        return merged_lines_all


def detect_lanes(imgDisplay):
    # convert to grayscale then black/white to binary image
    x = 700
    w = 1280
    y = 103
    h = 544
    frame = imgDisplay[y:y + h, x:x + w]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = 200
    frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("Black/White", frame)

    # blur image to help with edge detection
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # cv2.imshow("Blurred", blurred)

    # identify edges & show on screen
    edged = cv2.Canny(blurred, 30, 150)
    # cv2.imshow("Edged", edged)

    # perform full Hough Transform to identify lane lines
    # lines = cv2.HoughLines(edged, 1, np.pi / 180, 25)
    unmerged_lines = cv2.HoughLinesP(
        edged,
        rho=6,
        theta=np.pi / 60,
        threshold=150,
        lines=np.array([]),
        minLineLength=60,
        maxLineGap=5
    )
    global emergency_lane_lines
    if unmerged_lines is None:
        emergency_lane_lines = []
        return
    merger = HoughBundler()
    lines = merger.process_lines(unmerged_lines)
    # print(lines)
    for line in lines:
        line[0][0] += x
        line[0][1] += y
        line[1][0] += x
        line[1][1] += y
        cv2.line(imgDisplay, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 0, 255), 3)
    # define arrays for left and right lanes
    emergency_lane_lines = lines

    # overlay semi-transparent lane outline on original


cartracker = cartrack.Tracker()


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
        for line in emergency_lane_lines:
            p3 = np.array([t_x_bar, t_y_bar])
            p1 = np.array([line[0][0], line[0][1]])
            p2 = np.array([line[1][0], line[1][1]])
            d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
            min_dist.append(abs(d))
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
        if min(min_dist) < 60:
            rectColor = (0, 0, 255)
            text = '{}{} '.format(type, fid) + 'Emergency Lane Driving'
            print('found emergency lane driving {}{}'.format(type, fid))
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
                        default=0.8)

    results = parser.parse_args(args)
    return (results.model,
            results.input,
            results.output,
            results.frame_interval,
            results.vehicle_threshold)


id = 0
carid = 0

emergency_lane_lines = []

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

    while True:

        r, img = cap.read()
        if r:
            # img = cv2.resize(img, (1280, 720))
            # if frame_count % (frame_interval*25) == 0:
            #     id -= int(len(tracker.faceTrackers))
            #     tracker.deleteAll()
            #     print('refreshing')
            if frame_count % (frame_interval * 3) == 0:
                cartracker.removeDuplicate()
                # seattracker.removeDuplicate()

            cartracker.deleteTrack(img)
            if frame_count % frame_interval == 0:
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
                    elif classes[i] == 4 and scores[i] > vehiclethres - 0.3:
                        box = boxes[i]
                        matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            carid += 1
                            cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i],
                                                   'Motorcycle')
                    elif classes[i] == 6 and scores[i] > vehiclethres:
                        box = boxes[i]
                        matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            carid += 1
                            cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i], 'Bus')
                    elif classes[i] == 8 and scores[i] > vehiclethres:
                        box = boxes[i]
                        matchedID = cartracker.getMatchId(img, (box[1], box[0], box[3], box[2]))
                        if matchedID is None:
                            carid += 1
                            cartracker.createTrack(img, (box[1], box[0], box[3], box[2]), str(carid), scores[i],
                                                   'Truck')

            detect_lanes(img)

            car, motor, bus, truck = drawTrackedVehicle(img)

            cv2.putText(img, 'Cars: ' + str(car), (0, 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            cv2.putText(img, 'Motor: ' + str(motor), (0, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 2)
            cv2.putText(img, 'Bus: ' + str(bus), (0, 39),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
            cv2.putText(img, 'Truck: ' + str(truck), (0, 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 2)

            out.write(img)
            frame_count += 1
            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        else:
            raise RuntimeError('No more frame')
