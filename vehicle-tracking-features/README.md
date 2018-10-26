### Install
```
pip3 install requirements.txt
```
### How to run it
For car detection and tracking
```
python3 detect.py
```
For car plate detection
```
python3 carplate.py
```

# Prerequisite for running detect.py  
```
1. Download Tensorflow model that in COCO-trained models. High COCO mAP would be better but very slow.  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  
2. Extract the trained-models in model folder.  
3. Put your video in video folder and change detect.py video_src to your video path.
4. python3 detect.py
```
File option.py can allow some feature stated below. 

Feature:
Display vehicle type: set self.type = True  
Display vehicle color (although not accuracy): set self.color = True  
Display vehicle speed: set self.speed = True  
Display vehicle stay in a place: set self.stay = True  
Toll vehicle checking: set self.toll = True  
Display current total number of vehicle in current frame: set self.countTraffic = True  

# Prerequisite for running carplate.py  
(However the algorithm still need a lot of improvement)
```
1. setup aws credentials by running this command: aws configure  
2. Put your video in video folder and change carplate.py video_src to your video path.
```
