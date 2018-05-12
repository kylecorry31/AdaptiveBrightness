# Adaptive Brightness
Adds adaptive brightness to a computer without a light sensor. (Currently designed for my System76 Lemur, only designed to work on Linux and GNOME).

## Installation
Please install the following dependencies:
* Python 2.7 (python)
* OpenCV for Python (python-opencv)
* Numpy for Python (python-numpy)
* Psutil for Python (python-psutil)
* V4L-Utils (v4l-utils)
* Python tensorflow (pip install tensorflow)

Clone this repository and cd inside the folder.

It is also recommended to put a piece of a post-it note over the webcam, or else it may be tricked by white objects into thinking it is actually bright out.

## Running
Run the following command from this directory to start the adaptive brightness control.
```bash
python adaptive_brightness.py
```
