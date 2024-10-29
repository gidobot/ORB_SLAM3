cmake_minimum_required(VERSION 2.8.3)
project(viso2)

# find_package(catkin REQUIRED COMPONENTS
#   cv_bridge
# )

# add_definitions(-DPROFILING)

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g ")
# set(CUDA_ARCH "-arch=sm_72")  # Replace sm_75 with your architecture (e.g., sm_60, sm_80, etc.) sm_72 for jetson xavier nx
set(CUDA_ARCH "-arch=sm_86")  # Replace sm_75 with your architecture (e.g., sm_60, sm_80, etc.) sm_72 for jetson xavier nx
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH})
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++17")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-lineinfo") # Enable line info for debugging
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O2")

# Check if the system processor is ARM and set compiler flags accordingly
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64.*")
    message(STATUS "Detected ARM architecture")
    
    # Check if the target is 64-bit ARM (AArch64) and enable NEON
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(STATUS "Detected 64-bit ARM architecture (AArch64)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+simd")
    else()
        message(STATUS "Detected 32-bit ARM architecture")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon")
    endif()

    # Define __ARM_NEON__ macro globally
    add_definitions(-D__ARM_NEON__)
    message(STATUS "__ARM_NEON__ defined")
else()
    message(STATUS "Non-ARM architecture detected")
endif()

find_package(OpenCV REQUIRED
  COMPONENTS
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_xfeatures2d
#   PATHS ~/workspace/lib/opencv/build
#   NO_DEFAULT_PATH
#   CONFIG
)

MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

set(viso2_sources
  src/filter.cpp
  src/matcher.cpp
  src/matcher_sift.cpp
  src/matrix.cpp
  src/reconstruction.cpp
  src/triangle.cpp
  src/viso.cpp
  src/viso_mono.cpp
  src/viso_mono_omnidirectional.cpp
  src/viso_stereo.cpp
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(libviso2/src
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/../CudaSift
)

add_library(viso2 SHARED
  ${viso2_sources}
)

set_target_properties(viso2 PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"         
)

target_link_libraries (viso2 ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/../CudaSift/lib/libcudasift.so)