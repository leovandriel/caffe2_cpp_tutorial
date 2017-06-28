cmake_minimum_required(VERSION 2.6)

project (caffe2_cpp_tutorial)

find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cURL REQUIRED)

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

find_library(CAFFE2_LIB Caffe2_CPU)
find_library(GLOG_LIB glog)
find_library(GFLAGS_LIB gflags)

if(NOT CAFFE2_LIB)
  message(FATAL_ERROR "Caffe2 lib not found")
endif()

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

add_executable(pretrained src/pretrained.cpp)
target_link_libraries(pretrained ${CAFFE2_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB} ${GFLAGS_LIB} ${OpenCV_LIBS})

add_executable(mnist src/mnist.cpp)
target_link_libraries(mnist ${CAFFE2_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB} ${GFLAGS_LIB})

add_executable(imagenet src/imagenet.cpp)
target_link_libraries(imagenet ${CAFFE2_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB} ${GFLAGS_LIB} ${OpenCV_LIBS} ${CURL_LIBRARIES})

add_executable(retrain src/retrain.cpp)
target_link_libraries(retrain ${CAFFE2_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB} ${GFLAGS_LIB} ${OpenCV_LIBS} ${CURL_LIBRARIES})
