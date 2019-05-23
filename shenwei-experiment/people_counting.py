# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

# Import Libraries
import argparse
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
import track.tracker as track
import detector as od
import mysql.connector

tracker = track.Tracker()


# MySQL Connector 
# Change your username, password, database here
def db_connection():

	mydb = mysql.connector.connect(
		host="127.0.0.1",
		user="root",
		passwd="",
		database="people_tracking",
	)

	db_cursor = mydb.cursor(buffered=True, dictionary=True)
	return mydb, db_cursor


# Function to draw boxes on people
def drawTrackedPeople(imgDisplay):
    print(tracker.faceTrackers.keys()) # debug: display person id in session list
    for fid in tracker.faceTrackers.keys():
        print(fid) # display current person id in processing
        tracked_position = tracker.faceTrackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        timestayed = int(tracker.getTimeInSecond(fid))
		
        db_cursor.execute( "SELECT ID FROM people" )
        peoples = [item['ID'] for item in db_cursor.fetchall()]
		
        if int(fid) in peoples:
            print("fid is in list")
            db_cursor.execute( "UPDATE People SET session_time = {} WHERE ID = {};".format(timestayed, int(fid)) )
             
        else:
            print("fid is not in list")
            db_cursor.execute( "INSERT INTO People(session_time) VALUES ({});".format(timestayed) )

        mydb.commit()
        text = 'P{} '.format(fid) + str(timestayed) + 's'
        rectColor = (0, 255, 0)

        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        textX = int(t_x + t_w / 2 - (textSize[0]) / 2)
        textY = int(t_y)
        textLoc = (textX, textY)

        cv2.rectangle(imgDisplay, (t_x, t_y),
                      (t_x + t_w, t_y + t_h),
                      rectColor, 3)

        cv2.putText(imgDisplay, text, textLoc,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 1)

def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Script for counting person in zone detection')
    parser.add_argument('-m', '--model',
                        help='Tensorflow object detection model path',
                        default='model/frozen_inference_graph.pb')
    parser.add_argument('-i', '--input',
                        help='Input video filename',
                        default='rtsp://admin:tapway123@tapway1.dahuaddns.com/cam/realmonitor?channel=1&subtype=0')
                        #default='rtsp://admin:admin123@192.168.0.109/cam/realmonitor?channel=1&subtype=0')
                        #default='rtsp://admin:tapway123@tapway1.dahuaddns.com/cam/realmonitor?channel=1&subtype=0')
                        #default='test2.mp4')
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

    results = parser.parse_args(args)
    return (results.model,
            results.input,
            results.output,
            results.frame_interval,
            results.roi_points,
            results.people_threshold)



mydb, db_cursor = db_connection()
db_cursor.execute( "SELECT MAX(ID) FROM people")
id_query = db_cursor.fetchone()
print(id_query['MAX(ID)'])
if id_query['MAX(ID)'] == None:
    id = 0
else:
    id = id_query['MAX(ID)']

p1 = None
p2 = None

if __name__ == "__main__":
    model_path, input, output, frame_interval, points, threshold = check_arg(sys.argv[1:])
    frame_interval = int(frame_interval)
    threshold = float(threshold)
    points = eval(points)
    p1 = (points[0], points[1])
    p2 = (points[2], points[3])
    odapi = od.DetectorAPI(path_to_ckpt=model_path)
    cap = cv2.VideoCapture(input)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    flag, frame = cap.read()
    assert flag == True
    tracker.videoFrameSize = frame.shape
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
    out = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))
	
	
	### Comment these lines to disable preview
    cv2.namedWindow("Tapway - People Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tapway - People Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        r, img = cap.read()
        if r:
            if frame_count % (frame_interval * 5) == 0:
                tracker.removeDuplicate()

            tracker.deleteTrack(img)

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


            print("the id is " + str(id))
            drawTrackedPeople(img)
            number = int(len(tracker.faceTrackers))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            cv2.putText(img, 'People: ' + str(number), (0, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            out.write(img)
            frame_count += 1
            #imgS = cv2.resize(img, (768,432))
            cv2.imshow("Tapway - People Tracking", img) ### Comment this line to disable preview
            key = cv2.waitKey(60)
            if key & 0xFF == ord('q'):
                break
            # cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
        else:
            print('No more frame')
            break
            #raise RuntimeError('No more frame')

            
    cap.release()
    out.release()
    cv2.destroyAllWindows()   