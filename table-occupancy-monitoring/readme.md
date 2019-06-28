# Table Occupancy Monitoring
``` Status: In Development ```

## Introduction

<div>
  <p>This program is developed for table occupancy monitoring in a food court. The objective of this project is to count the dwell time of customers within a drawn region around a table. The result will be saved into a database. </p>
  <p>Link to trello card: https://trello.com/c/pJCN22We/130-ai-detect-number-of-people-and-how-much-time-spent-in-zones</p>
  <a style="float:left; padding:5px">
		<img src="https://github.com/tapway/tapway-ai-experiment/blob/master/table-occupancy-monitoring/sample/video_output.png" >
	</a>
</div>

## Prerequisites
#### 1) Install Anaconda (latest)
    Windows: https://repo.anaconda.com/archive/Anaconda3-2019.03-Windows-x86_64.exe 
    Linux: https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh 
    Full installation guide: https://docs.anaconda.com/anaconda/install/ 

#### 2) Setting up Anaconda
    To open Anaconda Navigator on Windows, find "Anaconda Navigator" on your computer, open it as usual.
    To open Anaconda Navigator on Linux: 
    - Open a terminal and use the following command
    - source anaconda3/bin/activate
    - anaconda-navigator

#### 3) Create an environment
    - On Windows/Linux, create a Python 3.6 Environment

#### 4) Install necessary packages with the following commands: (If there are still missing packages, please use conda install)
    - conda install -c anaconda tensorflow-gpu 
    - conda install -c conda-forge opencv
    - conda install -c conda-forge cmake
    - conda install -c conda-forge dlib
    - conda install -c anaconda mysql-connector-python
    - conda install -c anaconda cudatoolkit
    - conda install -c anaconda cudnn
  
#### 5) Setting up MySQL Database
    - database name = 'people_tracking'
    - database table = 'people'
    - attributes in 'people':
        => id int(11) NOT NULL
        => dwell_time int(11) NOT NULL
        => table_id int(11) NOT NULL

## Execution of the program
` Please always open a command prompt with your created environment to run the program! `

#### To execute the program for monitoring: 
    - Locate the project folder with your command prompt
    - "py people_counting.py" on Windows
    - "python3 people_counting.py" on Linux

#### To retrieve the ROI points of a region: 
    - "py get_roi_points.py" on Windows, "python3 get_roi_points.py" on Linux
    - Enter the name of the selected camera (cam0, cam1, cam2 ....)
    - Press on the windows that pops up, as the windows is not main focused when it appears 
    - Press "r"
    - Draw a box to locate the table (draw again if not satisfied)
    - Press "enter"
    - Press "q" to confirm
    - The ROI points are displayed on the command prompt
    - You may now insert the table_id and ROI points into configuration JSON file 
    (IMPORTANT: Remember to change the value of 'num_of_tables')

## Explanation on Detection and Tracking
#### Model and Dataset
    Model used for training is ""faster_rcnn_inception_v2_coco""
    Dataset used for training is "SCUT-HEAD-Dataset-Release"

#### Dataset Details
    We use Part A of SCUT-HEAD, 1500 images for training and 500 images for testing
    Number of steps: 39894
    
#### Training Tutorial
    A folder of images with head annotation has already been prepared for you in /sample/images/
    
## JSON Configuration
    Camera setting can be edited from config.json

## Resource Links
<p>How to train a model: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10</p>
<p>faster_rcnn_inception_v2_coco: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz</p>
<p>SCUT-HEAD-Dataset-Release: https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release</p>

## Important
Please download the video from https://www.youtube.com/watch?v=qOyWdqHsstk and save it as test.mp4 and put it into the project folder for testing. Make sure it is in `1080p`

You can also download it from https://studentmmuedumy-my.sharepoint.com/:v:/g/personal/1161304136_student_mmu_edu_my/EcDWb3AbgjlLtcyam-cBZWwBUlikVnU42SXwKc3trFBT5Q?e=VWDT2X
