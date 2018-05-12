# Adaptive Brightness
A personalized experience for screen brightness, built with machine learning algorithms that learn your brightness preferences in different surroundings, even if you don't have a light sensor.

Adds adaptive brightness to a computer without a light sensor by using your camera (just put a post-it note over it to diffuse the light). This program is designed for my System76 Lemur running Pop!\_OS 18.04, but it should run on any Linux computer running the GNOME desktop environment.

## Technology
Several modes will be available once the application is completed including a simple linear adaptive brightness (similar to most computers) and a machine learning based adaptive brightness (learns how the user likes the brightness set, like the feature found in Android P). Currently the machine learning algorithm is enabled by default, but a command line interface will be created to specify the algorithm used.

The machine learning based approach uses a simple neural network and supervised learning to learn how you adjust the brightness slider. When you notice that the brightness is too bright or dim for your preferences, adjust the slider and the system will learn to brighten or dim the screen in similar conditions

## Installation
Please install the following dependencies:
* Python 2.7 (python)
* OpenCV for Python (python-opencv)
* Numpy for Python (python-numpy)
* Psutil for Python (python-psutil)
* V4L-Utils (v4l-utils)
* Python tensorflow (pip install tensorflow)
* Python h5py (pip install h5py)

The following commands will install the dependencies for you:
```bash
sudo apt install -y python python-pip python-opencv python-numpy python-psutil v4l-utils
sudo python -m pip install tensorflow
sudo python -m pip install h5py
```

Then clone this repository
```bash
git clone https://github.com/kylecorry31/AdaptiveBrightness.git
```

It is also recommended to put a piece of a post-it note over the webcam, or else it may be tricked by white objects into thinking it is actually bright out.

## Running
Run the following command from this directory to start the adaptive brightness control.
```bash
cd AdaptiveBrightness
python adaptive_brightness.py
```
