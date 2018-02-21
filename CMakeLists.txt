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
include_directories(/home/xieyi/opt/caffe2/include)

find_library(CAFFE2_LIB caffe2)
find_library(CAFFE2_GPU_LIB caffe2_gpu)
find_library(DETECTRON_GPU_LIB caffe2_detectron_ops_gpu)
find_library(GLOG_LIB glog)
find_library(GFLAGS_LIB gflags)
find_library(GTEST_LIB gtest)
#find_library(NCCL_LIB nccl) # Somehow necessary to prevent "undefined reference to `ncclReduceScatter'"

set(PB_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/caffe2/util")
execute_process(
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --proto_path=../caffe2 --proto_path=${PB_SOURCE_DIR} --cpp_out=${PB_SOURCE_DIR} "${PB_SOURCE_DIR}/model.proto"
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC ")

if(NOT CAFFE2_LIB)
  message(FATAL_ERROR "Caffe2 lib not found")
endif()

set(ALL_LIBRARIES)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

macro(link_whole_archive lib)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    list(APPEND ALL_LIBRARIES -Wl,-force_load,$<TARGET_FILE_NAME:${lib}> ${lib})
  elseif(MSVC)
    list(APPEND ALL_LIBRARIES -WHOLEARCHIVE:$<TARGET_FILE_NAME:${lib}>)
  else()
    list(APPEND ALL_LIBRARIES -Wl,--whole-archive ${lib} -Wl,--no-whole-archive)
  endif()
endmacro()

file(GLOB LIB_SOURCES "${PROJECT_SOURCE_DIR}/src/caffe2/util/*.cc" "${PROJECT_SOURCE_DIR}/src/caffe2/operator/*.cc" "${PROJECT_SOURCE_DIR}/src/cvplot/*.cc")
add_library(caffe2_cpp ${LIB_SOURCES})
link_whole_archive(caffe2_cpp)

if(CUDA_LIBRARIES)
  # set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
  file(GLOB CUDA_SOURCES "${PROJECT_SOURCE_DIR}/src/caffe2/operator/*.cu")
  cuda_add_library(caffe2_cpp_gpu ${CUDA_SOURCES})
  link_whole_archive(caffe2_cpp_gpu)
endif()

list(APPEND ALL_LIBRARIES ${CAFFE2_LIB})

if(CAFFE2_GPU_LIB)
  list(APPEND ALL_LIBRARIES ${CAFFE2_GPU_LIB})
  add_definitions(-DWITH_GPU)
endif()

if(DETECTRON_GPU_LIB)
  list(APPEND ALL_LIBRARIES ${DETECTRON_GPU_LIB})
  ADD_DEFINITIONS(-DWITH_GPU)
endif()

list(APPEND ALL_LIBRARIES ${PROTOBUF_LIBRARY})
list(APPEND ALL_LIBRARIES ${GLOG_LIB} ${GFLAGS_LIB})
list(APPEND ALL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

if(OpenCV_LIBS)
  list(APPEND ALL_LIBRARIES ${OpenCV_LIBS})
  add_definitions(-DWITH_OPENCV)
endif()

if(CUDA_LIBRARIES)
  list(APPEND ALL_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY})
  add_definitions(-DWITH_CUDA)
endif()

if(CURL_LIBRARIES)
  list(APPEND ALL_LIBRARIES ${CURL_LIBRARIES})
  add_definitions(-DWITH_CURL)
endif()

if (NOT DEFINED CAFFE2_CPP_TUTORIAL_LIB_ONLY)
  file(GLOB BIN_SOURCES "${PROJECT_SOURCE_DIR}/src/caffe2/binaries/*.cc")
  foreach(filename ${BIN_SOURCES})
    get_filename_component(name ${filename} NAME_WE)
    add_executable(${name} ${filename})
    target_link_libraries(${name} ${ALL_LIBRARIES})
  endforeach()

  if(GTEST_LIB)
    file(GLOB TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/caffe2/util/*_test.cc" "${PROJECT_SOURCE_DIR}/test/caffe2/core/*_test.cc")
    foreach(filename ${TEST_SOURCES})
      get_filename_component(name ${filename} NAME_WE)
      add_executable(${name} ${filename})
      target_link_libraries(${name} ${ALL_LIBRARIES} ${GTEST_LIB})
    endforeach()
  endif()
endif()
