include(CMakeParseArguments)

# pybind_extension()
#
# Parameters:
# NAME: name of module
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# pybind_extension(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
#

# 首先尝试常规方式查找 pybind11
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
  # 如果没有找到 pybind11，使用 Python 脚本获取目录
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" "-c"
      "import pybind11; print(pybind11.get_include(), end='')"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    RESULT_VARIABLE PYBIND11_FIND_RESULT
  )

  if(NOT PYBIND11_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find pybind11 using Python")
  endif()

#   # 设置 pybind11 包含目录
#   include_directories(${PYBIND11_INCLUDE_DIR})

  # 获取 pybind11 的 CMake 目录
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" "-c"
      "import pybind11; print(pybind11.get_cmake_dir(), end='')"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    RESULT_VARIABLE PYBIND11_FIND_RESULT
  )

  if(NOT PYBIND11_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find pybind11 CMake directory using Python")
  endif()

  # 设置 pybind11_DIR 变量
  set(pybind11_DIR ${PYBIND11_CMAKE_DIR})

  # 再次尝试查找 pybind11
  find_package(pybind11 REQUIRED)
else()
  # 如果找到了 pybind11，确保包含目录被正确设置
#   include_directories(${pybind11_INCLUDE_DIRS})
endif()
include_directories(${pybind11_INCLUDE_DIRS})

if(NOT DEFINED PYTHON_MODULE_EXTENSION OR NOT DEFINED PYTHON_MODULE_DEBUG_POSTFIX)
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" "-c"
      "import sys, importlib; s = importlib.import_module('distutils.sysconfig' if sys.version_info < (3, 10) else 'sysconfig'); print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO'))"
    OUTPUT_VARIABLE _PYTHON_MODULE_EXT_SUFFIX
    ERROR_VARIABLE _PYTHON_MODULE_EXT_SUFFIX_ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(_PYTHON_MODULE_EXT_SUFFIX STREQUAL "")
    message(
      FATAL_ERROR "pybind11 could not query the module file extension, likely the 'distutils'"
                  "package is not installed. Full error message:\n${_PYTHON_MODULE_EXT_SUFFIX_ERR}"
    )
  endif()

  # This needs to be available for the pybind11_extension function
  if(NOT DEFINED PYTHON_MODULE_DEBUG_POSTFIX)
    get_filename_component(_PYTHON_MODULE_DEBUG_POSTFIX "${_PYTHON_MODULE_EXT_SUFFIX}" NAME_WE)
    set(PYTHON_MODULE_DEBUG_POSTFIX
        "${_PYTHON_MODULE_DEBUG_POSTFIX}"
        CACHE INTERNAL "")
  endif()

  if(NOT DEFINED PYTHON_MODULE_EXTENSION)
    get_filename_component(_PYTHON_MODULE_EXTENSION "${_PYTHON_MODULE_EXT_SUFFIX}" EXT)
    set(PYTHON_MODULE_EXTENSION
        "${_PYTHON_MODULE_EXTENSION}"
        CACHE INTERNAL "")
  endif()
endif()

function(pybind_extension)
  cmake_parse_arguments(
    PY # prefix
    "TESTONLY" # options
    "NAME" # one value args
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;LINKDIRS;DEPS" # multi value args
    ${ARGN}
  )

  if(PY_TESTONLY AND (NOT BUILD_TESTING))
    return()
  endif()

  add_library(${PY_NAME} SHARED)
  target_sources(${PY_NAME} 
    PRIVATE ${PY_SRCS} ${PY_HDRS}
  )
  target_link_libraries(${PY_NAME}
    PUBLIC ${PY_DEPS}
    PRIVATE ${PY_LINKOPTS}
  )
  # search directories for libraries
  target_link_directories(${PY_NAME}
    PUBLIC ${PY_LINKDIRS}
  )
  target_compile_options(${PY_NAME} PRIVATE ${PY_COPTS})
  target_compile_definitions(${PY_NAME} PUBLIC ${PY_DEFINES})

  # -fvisibility=hidden is required to allow multiple modules compiled against
  # different pybind versions to work properly, and for some features (e.g.
  # py::module_local). 
  if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
    set_target_properties(${PY_NAME} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
  endif()

  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
    set_target_properties(${PY_NAME} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
  endif()

  set_target_properties(
    ${PY_NAME}
    PROPERTIES PREFIX ""
               DEBUG_POSTFIX "${PYTHON_MODULE_DEBUG_POSTFIX}"
               SUFFIX "${PYTHON_MODULE_EXTENSION}")

endfunction()