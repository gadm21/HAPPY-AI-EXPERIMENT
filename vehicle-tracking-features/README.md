Download Tensorflow model that in COCO-trained models. High COCO mAP would be better but very slow.
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Main class - detect.py

File option.py can allow some feature stated below. 

Feature:
Display vehicle type: set self.type = True
Display vehicle color (although not accuracy): set self.color = True
Display vehicle speed: set self.speed = True
Display vehicle stay in a place: set self.stay = True
Toll vehicle checking: set self.toll = True
Display current total number of vehicle in current frame: set self.countTraffic = True