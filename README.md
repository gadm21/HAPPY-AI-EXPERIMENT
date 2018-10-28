# tapway-ai-experiment
Related to new features we are exploring for customers

### Server config
Base image - `Ubuntu Server 18.04 LTS (HVM), SSD Volume Type - ami-0c5199d385b432989`
##### list of installed packages:
`build-essential` `apt-transport-https` `ca-certificates` `curl` `software-properties-common` `docker-ce` `hstr` `byobu` `cmake` `awscli` `libsm6` `libxext6` `libxrender1` `xvfb` `libopencv-dev`

### Python dependences 
Due to libs build problems in different environments. `pip` replaced to [pipenv](https://pipenv.readthedocs.io/en/latest/)

##### If you need to create this environment locally do it this way:
```bash
pip install --user pipenv
git clone https://github.com/tapway/tapway-ai-experiment.git
cd tapway-ai-experiment
pipenv install
# check
 pipenv run python -V
# you must see something like 
# Courtesy Notice: Pipenv found itself running within a virtual environment, so it will automatically use that environment, instead of creating its own for any project. You can set PIPENV_IGNORE_VIRTUALENVS=1 to force pipenv to ignore that environment and create its own instead. You can set PIPENV_VERBOSITY=-1 to suppress this warning.
# Python 3.6.7
```
##### If you need to run script on our headless server, sequence is:
```bash
cd <path to repo>
pipenv shell
export DISPLAY=:1
python <what you want to run>
``` 
##### If you need to install new package do it this way:
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
