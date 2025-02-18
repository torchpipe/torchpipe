cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(USE_CCACHE "Attempt using CCache to wrap the compilation" OFF)
option(USE_CXX11_ABI "Use the new C++-11 ABI, which is not backwards compatible." OFF)
option(USE_MANYLINUX "Build for manylinux" OFF)
option(BUILD_TESTING "Build the test suite" OFF)
option(BUILD_PYBIND "Build the pybind11 extension" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MESSAGE_LOG_LEVEL STATUS)
set(CMAKE_VERBOSE_MAKEFILE ON)

if(POLICY CMP0135)
  # CMP0135: ExternalProject ignores timestamps in archives by default for the URL download method.
  cmake_policy(SET CMP0135 NEW)
endif()

function(parse_make_options options prefix)
  foreach(option ${options})
    string(REGEX REPLACE "(-D|-)" "" option ${option})
    string(REPLACE "=" ";" option ${option})
    list(GET option 0 option_name)
    list(GET option 1 option_value)
    set(${prefix}_${option_name}
        ${option_value}
        PARENT_SCOPE)
  endforeach()
endfunction()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" 
      CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." 
      FORCE
  )
endif()

# Convert the bool variable to integer.
if(USE_CXX11_ABI)
  set(USE_CXX11_ABI 1)
  message(STATUS "Using the C++-11 ABI.")
else()
  set(USE_CXX11_ABI 0)
  message(STATUS "Using the pre C++-11 ABI.")
endif()

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
    if (DEFINED ENV{CCACHE_DIR})
      message(STATUS "Using CCACHE_DIR: $ENV{CCACHE_DIR}")
    endif()
  else()
    message(WARNING "Could not find ccache. Consider installing ccache to speed up compilation.")
  endif()
endif()

# if defined, create and use the default binary cache for vcpkg
if (DEFINED ENV{VCPKG_DEFAULT_BINARY_CACHE})
  file(MAKE_DIRECTORY $ENV{VCPKG_DEFAULT_BINARY_CACHE})
  message(STATUS "Using VCPKG_DEFAULT_BINARY_CACHE: $ENV{VCPKG_DEFAULT_BINARY_CACHE}")
endif()

if (DEFINED ENV{DEPENDENCES_ROOT})
  message(STATUS "Using DEPENDENCES_ROOT: $ENV{DEPENDENCES_ROOT}")
endif()

 
 

# configure vcpkg
# have to set CMAKE_TOOLCHAIN_FILE before first project call.
if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
  message(STATUS "VCPKG_ROOT found, using vcpkg at $ENV{VCPKG_ROOT}")
else()
  include(FetchContent)
  if (DEFINED ENV{DEPENDENCES_ROOT})
    set(VCPKG_SOURCE_DIR $ENV{DEPENDENCES_ROOT}/vcpkg-src)
    message(STATUS "Using DEPENDENCES_ROOT: $ENV{DEPENDENCES_ROOT}")
  else()
    set(VCPKG_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/vcpkg-src)
    message(STATUS "Using FETCHCONTENT_BASE_DIR = ${FETCHCONTENT_BASE_DIR}")
  endif()

  if (USE_CXX11_ABI)
    FetchContent_Declare(vcpkg
      GIT_REPOSITORY "https://github.com/microsoft/vcpkg.git"
      GIT_TAG "2024.02.14"
      SOURCE_DIR ${VCPKG_SOURCE_DIR}
    )
  else()
    FetchContent_Declare(vcpkg
      GIT_REPOSITORY "https://github.com/vectorch-ai/vcpkg.git"
      GIT_TAG "ffc42e97c866ce9692f5c441394832b86548422c" # disable cxx11_abi
      SOURCE_DIR ${VCPKG_SOURCE_DIR}
    )
    message(STATUS "Using custom vcpkg with cxx11_abi disabled")
  endif()
  FetchContent_MakeAvailable(vcpkg)

  message(STATUS "Downloading and using vcpkg at ${vcpkg_SOURCE_DIR}")
  set(CMAKE_TOOLCHAIN_FILE ${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake
      CACHE STRING "Vcpkg toolchain file")
endif()

project(
  "hami"
  LANGUAGES C CXX  #CUDA
)

# find_package(CUDAToolkit REQUIRED)

# setup CMake module path, defines path for include() and find_package()
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# include custom cmake modules
include(static_analyzers)
# TODO: can't use sanitizers with CUDA for now.
# include(sanitizers)

if(UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
endif()

# find_package(Boost REQUIRED)
# find_package(Threads REQUIRED)
# # find all dependencies from vcpkg
# find_package(fmt CONFIG REQUIRED GLOBAL)
# find_package(glog CONFIG REQUIRED)
# find_package(gflags CONFIG REQUIRED)
# find_package(absl CONFIG REQUIRED)
# find_package(Protobuf CONFIG REQUIRED)
# find_package(gRPC CONFIG REQUIRED)
# find_package(re2 CONFIG REQUIRED)
# find_package(folly CONFIG REQUIRED)
if (BUILD_TESTING)
    # 查找 GTest
    find_package(GTest CONFIG QUIET)

    if (NOT GTest_FOUND)
        message(WARNING "GTest not found. Disabling tests.")
        set(BUILD_TESTING OFF)  # 禁用测试
    else()
        # 启用测试
        include(CTest)
        include(GoogleTest)
    endif()
endif()
# find_package(benchmark CONFIG REQUIRED)
# find_package(nlohmann_json CONFIG REQUIRED)
# find_package(prometheus-cpp CONFIG REQUIRED)
# find_package(RapidJSON CONFIG REQUIRED)

if (USE_MANYLINUX)
  # manylinux doesn't ship Development.Embed
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
else()
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
endif()

# find_package(NCCL REQUIRED)

if (USE_CXX11_ABI)
  # only use jemalloc if using the new C++-11 ABI
  find_package(Jemalloc)
  if(Jemalloc_FOUND)
    link_libraries(Jemalloc::jemalloc)
  endif()
endif()

# Important Note: Always invoke find_package for other dependencies
 # unexpected linker errors.


if (USE_CXX11_ABI)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_compile_definitions(USE_CXX11_ABI=1)

else()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_compile_definitions(USE_CXX11_ABI=0)

endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
 


# include current path
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

include_directories(${COMMON_INCLUDE_DIRS})

# add subdirectories
add_subdirectory(third_party)

add_subdirectory(cpp)

if(BUILD_PYBIND)
  add_subdirectory(hami/)
endif()
# add_subdirectory(scalellm)