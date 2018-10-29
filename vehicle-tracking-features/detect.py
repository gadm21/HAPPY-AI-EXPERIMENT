import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
import sys

from utils import label_map_util
from utils import visualization_utils as vis_util

import track.tracker as track
import color.image_processor as colorDetector
from PIL import Image
from PIL import ImageEnhance

import bisect
import vehicle as veh

import option

import argparse


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name("detection_boxes:0")

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name("detection_scores:0")
    classes = detection_graph.get_tensor_by_name("detection_classes:0")
    num_detections = detection_graph.get_tensor_by_name("num_detections:0")

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded},
    )

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
    )

    return image_np


def detect(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name("detection_boxes:0")

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name("detection_scores:0")
    classes = detection_graph.get_tensor_by_name("detection_classes:0")
    num_detections = detection_graph.get_tensor_by_name("num_detections:0")

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded},
    )

    return (np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))


def drawTrackedObject(imgDisplay):
    for oid in tracker.objectTrackers.keys():
        tracked_position = tracker.objectTrackers[oid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        rectColor = (255, 0, 0)

        class_name = tracker.vehicleList[oid].vehicleType
        color = tracker.vehicleList[oid].color
        speed = tracker.vehicleList[oid].speed
        stay = tracker.vehicleList[oid].stay

        text = "Type: " + class_name
        textColor = "Color: " + str(color)
        if speed is None:
            textSpeed = "Speed: None"
        else:
            textSpeed = "Speed: " + "{0:.2f}".format(speed)
        textStay = "Stay: " + str(stay)

        if opt.stay and stay >= 30:
            rectColor = (0, 0, 255)  # alert

        cv2.rectangle(imgDisplay, (t_x, t_y), (t_x + t_w, t_y + t_h), rectColor, 2)

        cv2.rectangle(
            imgDisplay, (t_x, t_y - opt.getTrueOption()), (t_x + 80, t_y), rectColor, -1
        )

        counter = 0
        if opt.type:
            cv2.putText(
                imgDisplay,
                text,
                (t_x, t_y - counter),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            counter += 10

        if opt.color:
            cv2.putText(
                imgDisplay,
                textColor,
                (t_x, t_y - counter),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            counter += 10

        if opt.speed:
            cv2.putText(
                imgDisplay,
                textSpeed,
                (t_x, t_y - counter),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            counter += 10

        if opt.stay:
            cv2.putText(
                imgDisplay,
                textStay,
                (t_x, t_y - counter),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            counter += 10


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="video path/ip address")
    parser.add_argument(
        "--width", default=640, type=int, help="the width of video output"
    )
    parser.add_argument(
        "--height", default=480, type=int, help="the height of video output"
    )
    parser.add_argument(
        "--output", default="output/output.mp4", type=str, help="output video path"
    )
    parser.add_argument(
        "--model",
        default="ssd_mobilenet_v1_coco_2018_01_28",
        type=str,
        help="model name folder in model",
    )
    return parser.parse_args(argv)


def main(args):
    if args.video is None:
        video_src = 0
    else:
        video_src = args.video
        # video_src = 'rtsp://admin:admin123@tapway2.dahuaddns.com/cam/realmonitor?channel=1&subtype=0'

    width = args.width
    height = args.height
    output = args.output
    MODEL_NAME = args.model

    video_capture = cv2.VideoCapture(video_src)

    frame_count = 0
    frame_interval = 10

    currentTrackID = 0

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    # out = cv2.VideoWriter(output,fourcc,20.0,(640,480))
    first = True
    num = 0
    pcount = 0  ## toll count
    ccount = 0
    status = None

    opt = option.option()

    tracker = track.Tracker()

    CWD_PATH = os.getcwd()

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = os.path.join(
        CWD_PATH, "model", MODEL_NAME, "frozen_inference_graph.pb"
    )

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(CWD_PATH, "label", "mscoco_label_map.pbtxt")

    NUM_CLASSES = 90

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)

    vehicleClass = ["bicycle", "car", "motorcycle", "bus", "truck"]

    while True:  # fps._numFrames < 120
        num += 1
        if opt.debug:
            print(num)
        flags, frame = video_capture.read()
        
        if flags == False:
            break        
        frame = cv2.resize(frame, (width, height))
        imgt = Image.fromarray(frame.copy())

        converter = ImageEnhance.Color(imgt)
        imgt = converter.enhance(1)

        frame = np.asarray(imgt.copy())

        im_height, im_width, _ = frame.shape
        tracker.videoFrameSize = frame.shape

        if first:
            out = cv2.VideoWriter(
                output, fourcc, opt.outputVideoFPS, (im_width, im_height)
            )
            first = False

        tracker.deleteTrack(frame.copy())

        if (frame_count % frame_interval) == 0:
            t = time.time()

            # output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
            output_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # output_rgb = detect_objects(output_rgb, sess, detection_graph)
            # print(detect(output_rgb,sess,detection_graph))
            (boxes, scores, classes) = detect(output_rgb, sess, detection_graph)

            for i in range(boxes.shape[0]):
                if scores[i] > 0.5:
                    box = boxes[i]
                    (ymin, xmin, ymax, xmax) = (
                        int(box[0] * im_height),
                        int(box[1] * im_width),
                        int(box[2] * im_height),
                        int(box[3] * im_width),
                    )

                    ### car does not so big ### just use for now
                    if ymax - ymin > 200 or xmax - xmin > 200:
                        continue

                    result = category_index[classes[i]]["name"]

                    if result not in vehicleClass:
                        continue

                    matchedID = tracker.getMatchId(
                        frame.copy(), (xmin, ymin, xmax, ymax)
                    )

                    if matchedID is None:
                        ccount += 1
                        vehicle = veh.Vehicle()
                        currentTrackID += 1

                        ### rgb image for color detector
                        img = output_rgb[ymin:ymax, xmin:xmax]

                        img = Image.fromarray(img)

                        # converter = ImageEnhance.Color(img)
                        # img = converter.enhance(0.5)

                        vehicle.color = colorDetector.process_image(img)

                        vehicle.vehicleType = category_index[classes[i]]["name"]

                        vehicle.positionX = (xmin + xmax) / 2.0
                        vehicle.positionY = (ymin + ymax) / 2.0

                        tracker.createTrack(
                            frame.copy(), (xmin, ymin, xmax, ymax), currentTrackID
                        )

                        tracker.vehicleList[currentTrackID] = vehicle

        drawTrackedObject(frame)

        if opt.toll:
            output_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            b1x1 = 0
            b1y1 = 360

            b1x2 = 80
            b1y2 = 380

            b2x1 = 250
            b2y1 = 410

            b2x2 = 330
            b2y2 = 430

            b3x1 = 490
            b3y1 = 465

            b3x2 = 580
            b3y2 = 485

            b1 = output_rgb[b1y1:b1y2, b1x1:b1x2]
            b2 = output_rgb[b2y1:b2y2, b2x1:b2x2]
            b3 = output_rgb[b3y1:b3y2, b3x1:b3x2]

            b11 = Image.fromarray(b1)
            b12 = Image.fromarray(b2)
            b13 = Image.fromarray(b3)

            ans1 = colorDetector.toll_process_image(b11)
            ans2 = colorDetector.toll_process_image(b12)
            ans3 = colorDetector.toll_process_image(b13)

            tcount = 0
            if ans1 == "Red" or ans1 == "White":
                tcount += 1
            if ans2 == "Red" or ans2 == "White":
                tcount += 1
            if ans3 == "Red" or ans3 == "White":
                tcount += 1
                # if tcount == 1 and status == 'close':
                # 	status == 'close'
            if tcount <= 1 and status == "close":
                pcount += 1
            status = "open"
            if tcount > 1:
                status = "close"
                # if tcount == 1 and status == 'close':
                # 	status == 'close'
            print(ans1, ans2, ans3)

            cv2.rectangle(frame, (b1x1, b1y1), (b1x2, b1y2), (255, 0, 0), 2)

            cv2.rectangle(frame, (b2x1, b2y1), (b2x2, b2y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (b3x1, b3y1), (b3x2, b3y2), (255, 0, 0), 2)

            text = "Toll: " + status
            cv2.putText(
                frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2
            )
            payment = "Payment: " + str(pcount)
            cv2.putText(
                frame, payment, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2
            )

            traffic = "Vehicle: " + str(ccount)
            cv2.putText(
                frame, traffic, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2
            )

        if opt.countTraffic:
            vehicleCount = len(tracker.objectTrackers)
            text = "Traffic : " + str(vehicleCount)
            cv2.putText(
                frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2
            )

        out.write(frame)

        cv2.imshow("Video", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = option.option()

    tracker = track.Tracker()

    CWD_PATH = os.getcwd()

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    MODEL_NAME = "ssd_mobilenet_v1_coco_2018_01_28"

    PATH_TO_CKPT = os.path.join(
        CWD_PATH, "model", MODEL_NAME, "frozen_inference_graph.pb"
    )

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(CWD_PATH, "label", "mscoco_label_map.pbtxt")

    NUM_CLASSES = 90

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)

    vehicleClass = ["bicycle", "car", "motorcycle", "bus", "truck"]

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

        sess = tf.Session(graph=detection_graph)

    main(parse_arguments(sys.argv[1:]))
