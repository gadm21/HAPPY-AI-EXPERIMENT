# tapway-ai-experiment
Related to new features we are exploring for customers

# Caveat
Due to libs build problems in different environments. `pip` replaced to [pipenv](https://pipenv.readthedocs.io/en/latest/)
1. [install](https://pipenv.readthedocs.io/en/latest/install/#installing-pipenv) `pipenv`
2. create venv(only once):
```bash
cd <project dir>
pipenv install
```
3. activate (every time then you need it): `pipenv shell`
4. If you need to install new package do it this way:
```
cd <project dir>
pipenv shell
pipenv install <package name>
# after that you need to commit Pipfile and Pipfile.lock to repo
```

# Source code

Link: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

We are using:
- faster rcnn inception - http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
- faster rcnn resnet50 - http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
- ssd inceptionv2 - http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz

Download the files and place them in the `\people-tracking-features\model` folder

# Useful links:
[How to use pipenv in your python project](https://jcutrer.com/howto/dev/python/pipenv-pipfile)
