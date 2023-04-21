# Embedded Object Detection

Simple C++ program used to detect objects in images using OpenCV and YOLOv7-tiny.

## Description

`object_detect.cpp` is a C++ program leveraging OpenCV and the Darknet opensource neural network for C libraries in conjunction with the YOLOv7-tiny object detection model to detect objects in live video feeds. The program takes in an image stream and outputs a new image steram with the detected objects outlined in a bounding box with included label and confidence level.

The future state of the program will be used to emit events to a ZeroMQ message queue for consumption by an AWS IoT Core integration service.

## Getting Started

## Prerequisites
Install the following dependencies and build tools on your system:
```
$ sudo apt install build-essential cmake git pkg-config libgtk-3-dev
$ sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
$ sudo apt install libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev
$ sudo apt install python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev
```

### Dependencies
#
#### OpenCV

* Checkout openCV with contrib modules
```
$ cd ~
$ mkdir opencv_with_contrib
$ cd opencv_with_contrib
$ git clone https://github.com/opencv/opencv.git
$ git clone https://github.com/opencv/opencv_contrib.git
```
* Checkout 4.7.0 branch of opencv
```
$ cd opencv
$ git checkout 4.1.1
$ cd ../opencv_contrib
$ git checkout 4.1.1
$ cd ..
```

* Make the build directory
```
$ mkdir build
$ cd build
```

* Build the project
```
$ cmake 
\-DCMAKE_BUILD_TYPE=RELEASE 
\-DOPENCV_ENABLE_NONFREE=ON 
\-DENABLE_PRECOMPILED_HEADERS=OFF 
\-DOPENCV_EXTRA_MODULES_PATH=~/opencv_with_contrib/opencv_contrib/modules 
\-DBUILD_opencv_legacy=OFF -DWITH_QT=ON 
\-DCMAKE_INSTALL_PREFIX=/usr/local ../opencv
$ make -j5
$ make install
```
Note you may need to install with `sudo` if you get permission errors.

#### [Darknet](https://pjreddie.com/darknet/)
* Clone the darknet repo
```
$ cd <WORKSPACE_ROOT>
$ git clone https://github.com/AlexeyAB/darknet
```
* Build the project
```
mkdir build_release
cd build_release
cmake ..
cmake --build . --target install --parallel 8
```

### Installing & Executing program

* Install above dependencies and clone this repo
* Build the program with the following command using the VSCode task labeled `Build`
* Alternatively, run the program with the debug launch configuration `Debug object_detect` in VSCode which will automatically build the project with debug hooks.

## Authors

Contributors names and contact info

Nate Brandeburg
[nate3d.com](https://www.nate3d.com/)

## Version History

* 0.1
    * Initial Release
    * Object detection using YOLOv7-tiny model
    * OpenCV integration
    * Darknet integration
    * ZeroMQ integration

## License

TBD
