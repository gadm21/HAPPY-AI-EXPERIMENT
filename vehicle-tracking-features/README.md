1. Download Tensorflow model that in COCO-trained models. High COCO mAP would be better but very slow.  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  
2. Put the trained-models in model folder.  
3. Put your video in video folder and change detect.py video_src to your video path.  
  
# detect.py - car detection and tracking  

File option.py can allow some feature stated below. 

Feature:
Display vehicle type: set self.type = True  
Display vehicle color (although not accuracy): set self.color = True  
Display vehicle speed: set self.speed = True  
Display vehicle stay in a place: set self.stay = True  
Toll vehicle checking: set self.toll = True  
Display current total number of vehicle in current frame: set self.countTraffic = True  

# carplate.py - detect car plate by using aws rekognition service  
1. setup aws credentials by running this command: aws configure  
2. Put your video in video folder and change license.py video_src to your video path.  
