cmake_minimum_required(VERSION 2.6)

project (caffe2_cpp_tutorial)

find_package(Protobuf REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cURL QUIET)
find_package(CUDA QUIET)
find_package(Threads)

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})
if(OpenCV_LIBS)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()
if(CUDA_LIBRARIES)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

find_library(CAFFE2_LIB Caffe2_CPU)
find_library(CAFFE2_GPU_LIB Caffe2_GPU)
find_library(GLOG_LIB glog)
find_library(GFLAGS_LIB gflags)
find_library(NCCL_LIB nccl)
find_library(CURAND_LIB curand)

if(NOT CAFFE2_LIB)
  message(FATAL_ERROR "Caffe2 lib not found")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/include/res/imagenet_classes.h)
  file(DOWNLOAD https://raw.githubusercontent.com/bwasti/AICamera/master/app/src/main/cpp/classes.h ${CMAKE_SOURCE_DIR}/include/res/imagenet_classes.h)
endif()

set(ALL_LIBRARIES ${ALL_LIBRARIES} ${CAFFE2_LIB})
set(ALL_LIBRARIES ${ALL_LIBRARIES} ${PROTOBUF_LIBRARY})
set(ALL_LIBRARIES ${ALL_LIBRARIES} ${GLOG_LIB} ${GFLAGS_LIB})
set(ALL_LIBRARIES ${ALL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if(OpenCV_LIBS)
  set(ALL_LIBRARIES ${ALL_LIBRARIES} ${OpenCV_LIBS})
  add_definitions(-DWITH_OPENCV)
endif()

if(CUDA_LIBRARIES)
  set(ALL_LIBRARIES ${ALL_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} ${NCCL_LIB} ${CURAND_LIB})
  add_definitions(-DWITH_CUDA)
endif()

if(CURL_LIBRARIES)
  set(ALL_LIBRARIES ${ALL_LIBRARIES} ${CURL_LIBRARIES})
  add_definitions(-DWITH_CURL)
endif()

if(CAFFE2_GPU_LIB)
  set(ALL_LIBRARIES ${ALL_LIBRARIES} ${CAFFE2_GPU_LIB})
  add_definitions(-DWITH_GPU)
endif()

file(GLOB LIB_SOURCES "src/caffe2/util/*.cc")
add_library(caffe2_cpp ${LIB_SOURCES})
set(ALL_LIBRARIES ${ALL_LIBRARIES} caffe2_cpp)

file(GLOB OPERATOR_SOURCES "src/caffe2/operator/*.cc")

add_executable(intro src/intro.cpp)
target_link_libraries(intro ${ALL_LIBRARIES})

add_executable(toy src/toy.cpp)
target_link_libraries(toy ${ALL_LIBRARIES})

add_executable(pretrained src/pretrained.cpp)
target_link_libraries(pretrained ${ALL_LIBRARIES})

add_executable(mnist src/mnist.cpp)
target_link_libraries(mnist ${ALL_LIBRARIES})

add_executable(imagenet src/imagenet.cpp)
target_link_libraries(imagenet ${ALL_LIBRARIES})

add_executable(retrain src/retrain.cpp ${OPERATOR_SOURCES})
target_link_libraries(retrain ${ALL_LIBRARIES})

add_executable(dream src/dream.cpp ${OPERATOR_SOURCES})
target_link_libraries(dream ${ALL_LIBRARIES})

add_executable(train src/train.cpp ${OPERATOR_SOURCES})
target_link_libraries(train ${ALL_LIBRARIES})

add_executable(diff src/diff.cpp)
target_link_libraries(diff ${ALL_LIBRARIES})
