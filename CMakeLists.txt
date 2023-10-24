
cmake_minimum_required(VERSION 3.15)

message(FATAL_ERROR "This CMake file is temporarily unmaintained. Contributions are welcome.")

set(CUDA_HOME /usr/local/cuda)
set(CMAKE_CUDA_COMPILER ${CUDA_HOME}/bin/nvcc)

project(ipipe LANGUAGES CXX CUDA)
set(CUDA  REQUIRED)
# cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1       for compile_commands.json


# ---------------------------------------------------------------------------------------
# Set default build to release
# ---------------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

 
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" OR CMAKE_BUILD_TYPE STREQUAL "Release")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "set build type to release" FORCE)
set(CMAKE_CXX_FLAGS "-fPIC -O3 -Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -W -Wall -Wno-unused-parameter")
 else()
 
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "set build type to release" FORCE)
# set(CMAKE_CXX_FLAGS "-fPIC -O0 -Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -W -Wall -Wno-unused-parameter")
endif()

 
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()


set(CMAKE_CXX_EXTENSIONS OFF)
# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(USE_PYTHON ON)


if (USE_PYTHON)
  add_definitions(-DUSE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Development NumPy Interpreter)
  

    if(NOT Python3_FOUND AND Python3_Development_FOUND)
        #   find_package() will not abort the build if anything's missing.
        string(JOIN "\n" errmsg
            "  Python3 and/or development libs not found."
            "  - Python3_FOUND=${Python3_FOUND}"
            "  - Python3_Development_FOUND=${Python3_Development_FOUND}"
            )
        message(FATAL_ERROR ${errmsg})
    endif()
    include_directories(${Python3_INCLUDE_DIRS})
    message("USE_PYTHON: ${Python3_INCLUDE_DIRS}")
endif()


# for libtorch
# add_definitions(-D _GLIBCXX_USE_CXX11_ABI=1)


# cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..


if(USE_PYTHON)
  execute_process(COMMAND python3 -c "import torch; print(torch.__file__)"
    RESULT_VARIABLE COMMAND_RESULT
    OUTPUT_VARIABLE PYTORCH_INIT_FILE
    ERROR_VARIABLE PYTHON_ERROR)
  if(NOT ${COMMAND_RESULT} STREQUAL "0")
    message(FATAL_ERROR "Failed to find Python's torch module.\n${PYTHON_ERROR}")
  endif()
  get_filename_component(Torch_ROOT ${PYTORCH_INIT_FILE} DIRECTORY)
  set(Torch_DIR ${Torch_ROOT}/share/cmake/Torch )
  
  message(STATUS "Using Python's torch installation located at ${Torch_DIR}")
endif()

find_package(Torch REQUIRED)


option(WITH_OPENCV "Compile with OPENCV" ON)  # OFF not supported at this moment
option(WITH_TENSORRT "Compile with TENSORRT" ON) # OFF not supported at this moment
add_definitions(-DWITH_TENSORRT)
add_definitions(-DWITH_OPENCV)

if (WITH_OPENCV)
    SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
    if(NOT DEFINED OPENCV_DIR)
        message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
    endif()
    if (WIN32)
    find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build/ NO_DEFAULT_PATH)
    else ()
    find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
    endif ()
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()



#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75)
#endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

find_package(Torch REQUIRED)
if (NOT TORCH_FOUND)
    message(FATAL_ERROR "Torch not found.")
else ()
    message(STATUS "Find Torch VERSION ${Torch_VERSION},
    TORCH_CXX_FLAGS ${TORCH_CXX_FLAGS}
    TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS},
    TORCH_LIBRARIES ${TORCH_LIBRARIES}, ")
endif ()

message(STATUS "CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fopenmp -fPIC")

message(STATUS "${Python3_INCLUDE_DIRS}")

include_directories( ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                        "/usr/local/include/opencv4/"
                        CMAKE_CURRENT_SOURCE_DIR
                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/spdlog/include/
                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/dep_sort/
                        CMAKE_SOURCE_DIR
                        CMAKE_INCLUDE_CURRENT_DIR
                        )
link_directories(${CUDA_HOME}/lib64
                )


FILE(GLOB srcs  
    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/core/src/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/backend/src/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/backend/src_cuda/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/backend/src_cuda/*.cu
    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/schedule/src/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/pipeline/src/*.cpp )
LIST(APPEND sdk_cpp_srcs ${srcs})

# FILE(GLOB srcs_examples
#     ${CMAKE_CURRENT_LIST_DIR}/examples/*.cpp)
# LIST(APPEND sdk_cpp_srcs ${srcs_examples})

include_directories(${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/core/include/
                    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/backend/include/
                    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/schedule/include/
                    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/pipeline/include/
                    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc
                    ${CMAKE_CURRENT_LIST_DIR}/torchpipe/csrc/thirdpart/
                    ${CMAKE_CURRENT_LIST_DIR}/thirdparty/)

message("CMAKE_CXX_FLAGS: "${CMAKE_CXX_FLAGS})
message("CMAKE_BUILD_TYPE: "${CMAKE_BUILD_TYPE})

add_library(ipipe SHARED ${sdk_cpp_srcs})

include(CMakeDependentOption)

option(BUILD_TEST "Build shared libs." ON)
cmake_dependent_option(
    INSTALL_TEST "Install test binaries if BUILD_TEST is on" ON
    "BUILD_TEST" OFF)
# add_subdirectory(test/cpp)


target_link_libraries(ipipe PUBLIC nvjpeg ${TORCH_LIBRARIES} opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui 
                nvinfer nvonnxparser nppig nppc)
