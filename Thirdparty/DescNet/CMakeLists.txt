cmake_minimum_required(VERSION 2.6)

project(descnet)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DCOMPILEDWITHC${CMAKE_CXX_STANDARD})

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../tensorrt-cpp-api/cmake" ${CMAKE_MODULE_PATH})

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# TODO: Specify the path to TensorRT root dir
# if (NOT TensorRT_DIR)
#     set(TensorRT_DIR /usr/src/tensorrt)
# endif()
# # Use the correct version of CUDA
# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
# set(TensorRT_NVONNXPARSER_LIBRARY /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.8)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(
${PROJECT_SOURCE_DIR}/include 
${PROJECT_SOURCE_DIR}/../tensorrt-cpp-api/src
${PROJECT_SOURCE_DIR}/../tensorrt-cpp-api/include
${PROJECT_SOURCE_DIR}/../tensorrt-cpp-api/include/interfaces
${OpenCV_INCLUDE_DIRS}
${CUDA_INCLUDE_DIRS}
${TensorRT_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
src/descnet.cc
src/model.cc
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
${OpenCV_LIBS}
${TensorRT_LIBRARIES}
${CUDA_cudadevrt_LIBRARY}
${CUDA_LIBRARIES}
${PROJECT_SOURCE_DIR}/../tensorrt-cpp-api/lib/libtensorrt_cpp_api.so
)