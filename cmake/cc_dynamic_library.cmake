include(CMakeParseArguments)

# inspired by https://github.com/abseil/abseil-cpp
# cc_dynamic_library()
# CMake function to imitate Bazel's cc_dynamic_library rule.
#
# Parameters:
# NAME: name of target
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# cc_dynamic_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# cc_dynamic_library(
#   NAME
#     fantastic_lib
#   SRCS
#     "b.cc"
#   DEPS
#     :awesome
# )
#
function(cc_dynamic_library)
  cmake_parse_arguments(
    CC_DYNAMIC_LIB # prefix
    "TESTONLY" # options
    "NAME" # one value args
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS;INCLUDES" # multi value args
    ${ARGN}
  )

  if(CC_DYNAMIC_LIB_TESTONLY AND (NOT BUILD_TESTING))
    return()
  endif()

  # Check if this is a header only library
  set(_CC_SRCS "${CC_DYNAMIC_LIB_SRCS}")
  foreach(src_file IN LISTS _CC_SRCS)
    if(${src_file} MATCHES ".*\\.(h|inc|hpp)")
      list(REMOVE_ITEM _CC_SRCS "${src_file}")
    endif()
  endforeach()

  if(_CC_SRCS STREQUAL "")
    set(CC_DYNAMIC_LIB_IS_INTERFACE 1)
  else()
    set(CC_DYNAMIC_LIB_IS_INTERFACE 0)
  endif()

  if(NOT CC_DYNAMIC_LIB_IS_INTERFACE)
    add_library(${CC_DYNAMIC_LIB_NAME} SHARED)
    # set_target_properties(${CC_DYNAMIC_LIB_NAME} PROPERTIES OUTPUT_NAME "hami")
    string(REGEX REPLACE "^lib" "" LIB_NAME_WITHOUT_PREFIX ${CC_DYNAMIC_LIB_NAME})
    set_target_properties(${CC_DYNAMIC_LIB_NAME} PROPERTIES OUTPUT_NAME ${LIB_NAME_WITHOUT_PREFIX})

    set_target_properties(${CC_DYNAMIC_LIB_NAME} PROPERTIES
      LINK_FLAGS "-Wl,--enable-new-dtags,-rpath,\\$ORIGIN"
    )


    target_link_options(${CC_DYNAMIC_LIB_NAME} PRIVATE
        "-static-libstdc++" 
        "-static-libgcc"
      
      )
      # target_link_options(${CC_DYNAMIC_LIB_NAME} PRIVATE -Wl,--exclude-libs,ALL)

  
      # "-Wl,--no-as-needed"
      # "-Wl,--version-script=${CMAKE_SOURCE_DIR}/glibc_version.map"


    target_sources(${CC_DYNAMIC_LIB_NAME} 
      PRIVATE ${CC_DYNAMIC_LIB_SRCS} ${CC_DYNAMIC_LIB_HDRS})
    
      # see https://github.com/NVIDIA/TensorRT-LLM/blob/0d0583a639cb120f09ae4af50dd0722bdd60a5df/cpp/tensorrt_llm/CMakeLists.txt
    target_link_libraries(${CC_DYNAMIC_LIB_NAME}
    PRIVATE -Wl,--whole-archive  ${CC_DYNAMIC_LIB_DEPS} "-Wl,-no-whole-archive"
      PRIVATE ${CC_DYNAMIC_LIB_LINKOPTS}
    )

    target_include_directories(${CC_DYNAMIC_LIB_NAME}
      PUBLIC 
        "$<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>"
      PRIVATE
        ${CC_DYNAMIC_LIB_INCLUDES}
    )
    target_compile_options(${CC_DYNAMIC_LIB_NAME} PRIVATE ${CC_DYNAMIC_LIB_COPTS})
    target_compile_definitions(${CC_DYNAMIC_LIB_NAME} PUBLIC ${CC_DYNAMIC_LIB_DEFINES})

    # set_target_properties(${CC_DYNAMIC_LIB_NAME} PROPERTIES
    #                       C_VISIBILITY_PRESET hidden
    #                       VISIBILITY_INLINES_HIDDEN ON
    #                       CXX_VISIBILITY_PRESET hidden
    #                       )

  else()
    message(FATAL_ERROR, "Header only libraries are not supported by cc_dynamic_library. Consider using cc_library instead.")
  endif()

  # add alias for the library target
  add_library(:${CC_DYNAMIC_LIB_NAME} ALIAS ${CC_DYNAMIC_LIB_NAME})
endfunction()