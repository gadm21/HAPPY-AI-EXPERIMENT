import os
import cv2
import argparse
import sys
from PIL import Image

from util import getRectangle,validBoundingBox
from track.tracker import Tracker
from vehicle import Vehicle
from option import Option
from detector import Detector
from model import Model
from label import Label

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="video path/ip address")
    parser.add_argument(
        "--width", default=-1, type=int, help="the width of video output"
    )
    parser.add_argument(
        "--height", default=-1, type=int, help="the height of video output"
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

def drawTrackedObject(imgDisplay):
    for oid in tracker.objectTrackers.keys():
        tracked_position = tracker.objectTrackers[oid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        rectColor = (255, 0, 0)

        vehicle = tracker.vehicleList[oid]

        class_name = vehicle.vehicleType
        color = vehicle.color
        speed = vehicle.speed
        stay = vehicle.stay

        textType = "Type: " + class_name
        textColor = "Color: " + str(color)
        if speed==0:
            textSpeed = "Speed: None"
        else:
            if speed > 0 and speed < 10:
                rectColor = (0,0,255)
                textSpeed = "Anomaly"
            else:
                textSpeed = "Speed: " + "{0:.2f}".format(speed)
        textStay = "Stay: " + str(stay)


        if opt.stay and stay >= 1:
            rectColor = (0, 0, 255)  # alert

        textTuple = (textType,textColor,textSpeed,textStay)
        optTuple = opt.getOptionTuple()
        titleHeight = optTuple.count(True)*10

        # draw object bounding box
        cv2.rectangle(
            imgDisplay,
            (t_x, t_y),
            (t_x + t_w, t_y + t_h),
            rectColor,
            2
        )

        # fill object title background
        cv2.rectangle(
            imgDisplay,
            (t_x, t_y - titleHeight),
            (t_x + 80, t_y),
            rectColor,
            -1
        )

        # write text on object title part
        counter = 0
        for i in range(0,len(textTuple)):
            if optTuple[i]:
                cv2.putText(
                    imgDisplay,
                    textTuple[i],
                    (t_x, t_y - counter),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )                
                counter += 10

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    CWD_PATH = os.getcwd()

    MODEL_NAME = args.model

    PATH_TO_CKPT = os.path.join(
        CWD_PATH, "model", MODEL_NAME, "frozen_inference_graph.pb"
    )

    model = Model(PATH_TO_CKPT)
    model.setup()

    PATH_TO_LABELS = os.path.join(CWD_PATH, "label", "mscoco_label_map.pbtxt")

    label = Label(PATH_TO_LABELS)
    label.setup()

    detector = Detector()
    detector.setModel(model)

    opt = Option()
    opt.type = False
    opt.color = False
    opt.speed = True
    opt.stay = False
    
    tracker = Tracker()

    objectClass = ["bicycle", "car", "motorcycle", "bus", "truck"]

    if args.video is None:
        video_src = 0
    else:
        video_src = args.video

    cap = cv2.VideoCapture(video_src)

    resize = False

    if args.width > 0:
        width = args.width
        resize = True
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if args.height > 0:
        height = args.height
        resize = True
    else:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker.videoFrameSize = (height, width)
    print(height,width,'gg')
    output = args.output

    frame_count = 0
    frame_interval = 5

    currentTrackID = 0

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # this video fps have problem
    # fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30
    tracker.fps = fps
    print('fps',fps)
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    while True:  # fps._numFrames < 120

        flags, frame = cap.read()

        if flags == False:
            break

        if resize:
            frame = cv2.resize(frame, (width, height))
        imgDisplay = frame.copy()

        tracker.deleteTrack(frame.copy())

        if (frame_count % frame_interval) == 0:

            output_rgb = cv2.cvtColor(imgDisplay, cv2.COLOR_BGR2RGB)

            (boxes, scores, classes) = detector.detectObject(output_rgb)

            for i in range(boxes.shape[0]):
                rect = getRectangle(boxes[i], width, height)
                result = label.getLabel(classes[i])
                check = validBoundingBox(rect, scores[i], result, objectClass)

                if check is False:
                    continue

                matchedID = tracker.getMatchId(frame, rect)
                # if current detected object already have tracker, do nothing
                if matchedID is not None:
                    continue

                    # else create new tracker for the object
                (xmin, ymin, xmax, ymax) = rect

                vehicle = Vehicle()
                currentTrackID += 1

                ### rgb image for color detector
                img = output_rgb[ymin:ymax, xmin:xmax]

                img = Image.fromarray(img)

                vehicle.color = detector.detectColor(img)
                vehicle.vehicleType = result
                vehicle.centerX = (xmin + xmax) / 2.0
                vehicle.centerY = (ymin + ymax) / 2.0

                tracker.createTrack(frame, rect, currentTrackID)

                tracker.vehicleList[currentTrackID] = vehicle

        # updateZone(zones,tracker)
        drawTrackedObject(imgDisplay)
        # drawZone(imgDisplay, zones)
        
        ''' this use to draw line for illegal video
        cv2.line(imgDisplay,(280,200),(700,200),(0,255,0),5)
        cv2.line(imgDisplay,(0,500),(1030,500),(0,255,0),5)
        '''

        # cv2.line(imgDisplay,(700,400),(750,450),(255,0,0),5)

        out.write(imgDisplay)

        cv2.imshow("Video", imgDisplay)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()