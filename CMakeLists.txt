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

set(ALL_LIBRARIES)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/include/res/imagenet_classes.h)
  file(DOWNLOAD https://raw.githubusercontent.com/bwasti/AICamera/master/app/src/main/cpp/classes.h ${CMAKE_SOURCE_DIR}/include/res/imagenet_classes.h)
endif()

file(GLOB LIB_SOURCES "src/caffe2/util/*.cc" "src/caffe2/operator/*.cc")
add_library(caffe2_cpp ${LIB_SOURCES})
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND ALL_LIBRARIES -Wl,-force_load caffe2_cpp)
else()
  list(APPEND ALL_LIBRARIES -Wl,--whole-archive caffe2_cpp -Wl,--no-whole-archive)
endif()

if(CUDA_LIBRARIES)
  # set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
  list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
  file(GLOB CUDA_SOURCES "src/caffe2/operator/*.cu")
  cuda_add_library(caffe2_cpp_gpu ${CUDA_SOURCES})
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    list(APPEND ALL_LIBRARIES -Wl,-force_load caffe2_cpp_gpu)
  else()
    list(APPEND ALL_LIBRARIES -Wl,--whole-archive caffe2_cpp_gpu -Wl,--no-whole-archive)
  endif()
endif()

list(APPEND ALL_LIBRARIES ${CAFFE2_LIB})

if(CAFFE2_GPU_LIB)
  list(APPEND ALL_LIBRARIES ${CAFFE2_GPU_LIB})
  add_definitions(-DWITH_GPU)
endif()

list(APPEND ALL_LIBRARIES ${PROTOBUF_LIBRARY})
list(APPEND ALL_LIBRARIES ${GLOG_LIB} ${GFLAGS_LIB})
list(APPEND ALL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

if(OpenCV_LIBS)
  list(APPEND ALL_LIBRARIES ${OpenCV_LIBS})
  add_definitions(-DWITH_OPENCV)
endif()

if(CUDA_LIBRARIES)
  list(APPEND ALL_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} ${NCCL_LIB} ${CURAND_LIB})
  add_definitions(-DWITH_CUDA)
endif()

if(CURL_LIBRARIES)
  list(APPEND ALL_LIBRARIES ${CURL_LIBRARIES})
  add_definitions(-DWITH_CURL)
endif()

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

add_executable(retrain src/retrain.cpp)
target_link_libraries(retrain ${ALL_LIBRARIES})

add_executable(dream src/dream.cpp)
target_link_libraries(dream ${ALL_LIBRARIES})

add_executable(train src/train.cpp)
target_link_libraries(train ${ALL_LIBRARIES})

add_executable(diff src/diff.cpp)
target_link_libraries(diff ${ALL_LIBRARIES})

add_executable(inspect src/inspect.cpp)
target_link_libraries(inspect ${ALL_LIBRARIES})
