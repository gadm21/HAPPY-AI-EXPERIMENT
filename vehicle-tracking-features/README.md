### Install
```
pip3 install requirements.txt
```
### Prerequisite for running detect.py  
none  

### Prerequisite for running carplate.py  
(However the algorithm still need a lot of improvement)
```
1. setup aws credentials by running this command: aws configure  
```
### How to run it
For car detection and tracking
```
python3 detect.py --video 'rtsp://admin:admin123@tapway2.dahuaddns.com/cam/realmonitor?channel=1&subtype=0'
```
For car plate detection
```
python3 carplate.py --video 'video/A0026.mpg' --width 704 --height 576
```

### Use other trained models for running detect.py  
```
1. Download Tensorflow model that in COCO-trained models. High COCO mAP would be better but very slow.  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  
2. Extract the trained-models in model folder.  
3. python3 detect.py --video <video> --model <modelname>
```
### Detail about detect.py
File option.py can allow some feature stated below. 

Feature:
Display vehicle type: set self.type = True  
Display vehicle color (although not accuracy): set self.color = True  
Display vehicle speed: set self.speed = True  
Display vehicle stay in a place: set self.stay = True  
Toll vehicle checking: set self.toll = True  
Display current total number of vehicle in current frame: set self.countTraffic = True  
