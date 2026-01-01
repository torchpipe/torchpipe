# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

cmake_minimum_required(VERSION 3.18)
project(omniback)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TVM_FFI_EXT_SHIP_DEBUG_SYMBOLS "Ship debug symbols" ON)
option(USE_CXX11_ABI "Use C++11 ABI" ON)
option(BUILD_PY "build omniback pybind11 module" OFF)


if(USE_CXX11_ABI)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
else()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

set(USE_FFI ON)

if(USE_FFI)
    add_definitions(-DUSE_FFI)
else()
    message(FATAL_ERROR "USE_FFI must be set to ON at this moment")
endif()



find_package(tvm_ffi CONFIG REQUIRED)

# Core library: compile all cpp files in cpp/omniback/
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

add_library(omniback_core SHARED ${CORE_SOURCES})


add_library(omniback_header INTERFACE)
target_compile_features(omniback_core PRIVATE cxx_std_17)

target_include_directories(
  omniback_header INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>
)
target_include_directories(
  omniback_header INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spdlog/include>
                           $<INSTALL_INTERFACE:include>
)

target_include_directories(omniback_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spdlog/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/bs_thread_pool
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tinytoml
)

target_link_libraries(omniback_core 
    PUBLIC omniback_header
    PRIVATE tvm_ffi_header  
    PRIVATE tvm_ffi_shared  
)


# Set output name to libomniback.so (without lib prefix on Unix)
set_target_properties(omniback_core PROPERTIES 
    OUTPUT_NAME "omniback"
    PREFIX "lib"
    SUFFIX ".so"
)

if(BUILD_PY)
    # Find necessary packages
    find_package(
    Python
    COMPONENTS Interpreter Development.Module
    REQUIRED
    )
    find_package(pybind11 REQUIRED)  # Required for Python bindings

    # Python bindings: compile pybind files in omniback/pybind/
    file(GLOB_RECURSE PYTHON_SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/omniback/pybind/*.cpp"
    )
    pybind11_add_module(omniback_py MODULE ${PYTHON_SOURCES})

    target_compile_definitions(omniback_py
        PRIVATE PY_MODULE_NAME=omniback_py
    )
    target_compile_features(omniback_py INTERFACE cxx_std_17)

    # Configure Python module
    # target_include_directories(omniback_py PRIVATE
    #     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # )
    target_link_libraries(omniback_py PRIVATE
        omniback_core
        tvm_ffi_header
        tvm_ffi_shared
        omniback_header
        Python::Module
    )

    # # Set Python module output name to match expected import name
    # set_target_properties(omniback_py PROPERTIES 
    #     PREFIX ""
    #     SUFFIX ".${PYTHON_EXTENSION_SUFFIX}"
    # )
endif()

# Handle debug symbols for core library
if(TVM_FFI_EXT_SHIP_DEBUG_SYMBOLS)
    tvm_ffi_add_prefix_map(omniback_core ${CMAKE_CURRENT_SOURCE_DIR})
    tvm_ffi_add_apple_dsymutil(omniback_core)
    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
        DESTINATION .
        FILES_MATCHING
        PATTERN "*.dSYM"
    )
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/omniback/ DESTINATION include/omniback/)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spdlog/include/ DESTINATION include/)
install(TARGETS omniback_core DESTINATION lib)
if(BUILD_PY)
    install(TARGETS omniback_py DESTINATION .)
endif()