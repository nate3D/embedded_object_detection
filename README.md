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

## Dependencies
#
### OpenCV

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
$ git checkout 4.7.0
$ cd ../opencv_contrib
$ git checkout 4.7.0
$ cd ..
```

* Make the build directory
```
$ mkdir build
$ cd build
```

* Build the project
```
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr \
-D OPENCV_EXTRA_MODULES_PATH=/opt/whisker/opencv_with_contrib/opencv_contrib/modules \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D WITH_OPENCL=OFF \
-D WITH_CUDA=ON \
-D CUDA_ARCH_BIN=5.3 \
-D CUDA_ARCH_PTX="" \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_NEON=ON \
-D WITH_QT=ON \
-D WITH_OPENMP=ON \
-D BUILD_TIFF=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_TBB=ON \
-D BUILD_TBB=ON \
-D BUILD_TESTS=OFF \
-D WITH_EIGEN=ON \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=OFF \
-D MAKE_INSTALL_PREFIX=/usr/local \
../opencv
$ make -j4
$ make install
```
Note you may need to install with `sudo` if you get permission errors.
Note you will need 8GB swap space for building on Nvidia Jetson Nano dev board

### [ZeroMQ](https://github.com/zeromq/cppzmq)
*Build steps:
1. Build [libzmq](https://github.com/zeromq/libzmq) via cmake. This does an out of source build and installs the build files
   - download and unzip the lib, cd to directory
   - mkdir build
   - cd build
   - cmake ..
   - sudo make -j4 install

2. Build cppzmq via cmake. This does an out of source build and installs the build files
   - download and unzip the lib, cd to directory
   - mkdir build
   - cd build
   - cmake ..
   - sudo make -j4 install

3. Build cppzmq via [vcpkg](https://github.com/Microsoft/vcpkg/). This does an out of source build and installs the build files
   - git clone https://github.com/Microsoft/vcpkg.git
   - cd vcpkg
   - ./bootstrap-vcpkg.sh # bootstrap-vcpkg.bat for Powershell
   - ./vcpkg integrate install
   - ./vcpkg install cppzmq

### [AWS C++ SDK](https://docs.aws.amazon.com/sdk-for-cpp/v1/developer-guide/setup-linux.html)
* Clone the AWS C++ SDK repo
```
$ cd <WORKSPACE_ROOT>
$ git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp
```
* Build the project
```
mkdir build_release
cd build_release
cmake ../aws-sdk-cpp -DCMAKE_BUILD_TYPE=[Debug | Release] -DCMAKE_PREFIX_PATH=/usr/local/ -DCMAKE_INSTALL_PREFIX=/usr/local/-DBUILD_ONLY="iot;core;s3"
make -j8
make install
```
Note you may need to install with `sudo` if you get permission errors.

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
