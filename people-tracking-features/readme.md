# Scripts written for PLUS Project

# Prerequisite
  - All scripts require Python 3.6.2 or above.
  - Install the dependencies before running any scripts
  `pip install -r requirements.txt`

Each script is independent of each other and is written to comply with requirements by POC stated here https://docs.google.com/spreadsheets/d/1Bbt4CXrY_K-9glSev1PensSyB0vsUxiZhn9PvC5xaVY/edit#gid=1472470514

| Script Name | Requirement No. |
| ------ | ------ |
| [detect_unattended_package.py](detect_unattended_package.py) | A4c |
| [emergency_lane.py](emergency_lane.py) | B4 |
| [height_detection.py](height_detection.py) | C6 |
| [intrusion_detection.py](intrusion_detection.py) | C3 |
| [lighting.py](lighting.py) | C5 |
| [line_counting.py](line_counting.py) | A4a, A4b |
| [loitering.py](loitering.py) | A4b |
| [parking_spots.py](parking_spots.py) | A2b |
| [weather.py](weather.py) | B6 |
| [zone_count_dwell_seats.py](zone_count_dwell_seats.py) | A3a |

### Example Usage
`python detect_unattended_package.py -m model\frozen_inference_graph.pb -i input.mp4 -o output.avi`

`-m` parameter should be the path to the Tensorflow Object Detection frozen graph (*.pb file)
`-i` parameter is the path to video file for processing (any format that opencv supports)
`-o` parameter is the path to save the processed video file (*.avi file format) [optional]

Use `-h` for more information on each file's usage.

### Note
1. `gui` folder contains important tkinter library for user interface
2. `model` folder contains the inference graph of a self-trained model for `zone_count_dwell_seats.py`, but it can be used on other scripts as well.
3. `track` folder contains object tracker library created for each type of object that is being tracked.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
