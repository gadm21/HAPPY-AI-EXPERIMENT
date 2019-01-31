# Vehicle Counting for PLUS videos

# Prerequisite
  - All scripts require Python 3.6.2 or above.
  - tensorflow_p36 env on AWS EC2 Ubuntu DLAMI
  - OpenCV 3.4.5 or above

### Example Usage


`python vehicle-count.py -i videos/PBridge-KM100WB-realtime-8s.mp4 -o output.avi`



`-i` parameter is the path to video file for processing (any format that opencv supports)

`-m` [optional] parameter should be the path to the Tensorflow Object Detection frozen graph (*.pb file)
    default='model/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb' 

`-o` [optional] parameter is the path to save the processed video file (*.avi file format) default='output.avi'

`-f` [optional] Number of frame interval between frame processing, default=5

`-vt` [optional] Threshold value for vehicle detection',default=0.6 (-vt 0.4 or 0.5 can be used)


Use `-h` for more information on each file's usage.


### Note

1. `model` folder contains the inference graph of a pre-trained model for `*.py`.
3. `track` folder contains object tracker library created for object that is being tracked.



