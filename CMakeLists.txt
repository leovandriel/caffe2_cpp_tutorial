cmake_minimum_required(VERSION 2.6)

project (caffe2_cpp_tutorial)

find_package(opencv REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

find_library(CAFFE2_LIB Caffe2_CPU)
find_library(GLOG_LIB glog)
find_library(GFLAGS_LIB gflags)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

file(GLOB SOURCES "src/*.cpp")

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/include/res/imagenet_classes.h)
    file(DOWNLOAD https://raw.githubusercontent.com/bwasti/AICamera/master/app/src/main/cpp/classes.h ${CMAKE_SOURCE_DIR}/include/res/imagenet_classes.h)
endif()

add_executable(intro src/intro.cpp)
target_link_libraries(intro ${CAFFE2_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB})

add_executable(toy src/toy.cpp)
target_link_libraries(toy ${CAFFE2_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB})

add_executable(squeeze src/squeeze.cpp)
target_link_libraries(squeeze ${CAFFE2_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB} ${GFLAGS_LIB} ${OpenCV_LIBS})

add_executable(mnist src/mnist.cpp)
target_link_libraries(mnist ${CAFFE2_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB} ${GFLAGS_LIB})
