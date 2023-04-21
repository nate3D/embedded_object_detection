cmake_minimum_required(VERSION 3.0)
project(object_detect)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(cppzmq REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${cppzmq_INCLUDE_DIRS})

add_executable(object_detect src/object_detect.cpp)

target_link_libraries(object_detect ${OpenCV_LIBS} ${cppzmq_LIBS})
