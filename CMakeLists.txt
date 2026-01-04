# Copyright 2021-2023 NetEase.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)
project(omniback)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(SKIP_CMAKE)
    message(STATUS "SKIP_CMAKE is ON: Skipping native extension build")
    return()
endif()

# option(USE_CXX11_ABI "Use C++11 ABI (ignored if BUILD_BOTH_ABIS is ON)" ON)
option(BUILD_BOTH_ABIS "Build both C++11 and C++03 ABI versions of the library" ON)
 

option(SKIP_CMAKE "Skip cmake" OFF)

set(USE_CXX11_ABI_DEFAULT "1")  # fallback

# Only probe if env var not set
if(NOT DEFINED ENV{USE_CXX11_ABI})
    find_package(Python COMPONENTS Interpreter REQUIRED)
    if(Python_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" "-c"
                    "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
            OUTPUT_VARIABLE TORCH_ABI
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(TORCH_ABI MATCHES "^[01]$")
            set(USE_CXX11_ABI_DEFAULT "${TORCH_ABI}")
        endif()
    endif()
endif()

option(USE_CXX11_ABI "Use C++11 ABI (ignored if BUILD_BOTH_ABIS is ON)" ${USE_CXX11_ABI_DEFAULT})

message(STATUS "BUILD_BOTH_ABIS = ${BUILD_BOTH_ABIS}, USE_CXX11_ABI = ${USE_CXX11_ABI}, USE_CXX11_ABI_DEFAULT = ${USE_CXX11_ABI_DEFAULT}")



# --- Find dependencies ---
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE tvm_ffi_ROOT
)
find_package(tvm_ffi CONFIG REQUIRED)

# --- Collect sources (excluding pybind) ---
file(GLOB_RECURSE ALL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omniback/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omniback/builtin/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omniback/ffi/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omniback/helper/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omniback/schedule/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/omniback/py/*.cpp"
)
list(FILTER ALL_SOURCES EXCLUDE REGEX "src/omniback/pybind/")
set(CORE_SOURCES ${ALL_SOURCES})

# --- Header-only interface library ---
add_library(omniback_header INTERFACE)
target_include_directories(
    omniback_header INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spdlog/include>
)

# --- Helper function to create ABI-specific library ---
function(add_omniback_library TARGET_NAME ABI_FLAG OUTPUT_NAME_SUFFIX)
    add_library(${TARGET_NAME} SHARED ${CORE_SOURCES})

    if(${ABI_FLAG})
        target_compile_definitions(${TARGET_NAME} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    else()
        target_compile_definitions(${TARGET_NAME} PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
    endif()

    target_include_directories(${TARGET_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spdlog/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/bs_thread_pool
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tinytoml
    )

    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

    target_compile_definitions(${TARGET_NAME} PRIVATE USE_FFI)

    target_link_libraries(${TARGET_NAME} 
        PUBLIC omniback_header)

    tvm_ffi_configure_target(${TARGET_NAME} LINK_SHARED OFF LINK_HEADER ON)


    set_target_properties(${TARGET_NAME} PROPERTIES
        OUTPUT_NAME "omniback${OUTPUT_NAME_SUFFIX}"
        PREFIX "lib"
        SUFFIX ".so"
    )

    install(TARGETS ${TARGET_NAME} DESTINATION libs)
endfunction()

# --- Main logic: single vs both ABIs ---
if(BUILD_BOTH_ABIS)
    message(STATUS "Building both C++11 and C++03 ABI versions")
    add_omniback_library(omniback_core_cxx11   TRUE  "")
    add_omniback_library(omniback_core_cxx03   FALSE "_cxx03")
else()
    # Use USE_CXX11_ABI for single build
    if(USE_CXX11_ABI)
        message(STATUS "Building C++11 ABI version (USE_CXX11_ABI=ON)")
        add_omniback_library(omniback_core TRUE "")
    else()
        message(STATUS "Building C++03 ABI version (USE_CXX11_ABI=OFF)")
        add_omniback_library(omniback_core FALSE "_cxx03")
    endif()
endif()

# --- Install headers ---
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/omniback/ DESTINATION include/omniback/)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spdlog/include/ DESTINATION include/)

# tvm_ffi_install(omniback_core DESTINATION .)



install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/plugins/
    DESTINATION plugins
    FILES_MATCHING
    PATTERN "csrc/**/*.cpp"
    PATTERN "csrc/**/*.cc"
    PATTERN "csrc/**/*.hpp"
    PATTERN "csrc/**/*.h"
    # PATTERN "examples" EXCLUDE
)