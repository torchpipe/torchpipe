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
  
  add_compile_options(-fvisibility=default)  # Linux/macOS

  if(NOT CC_DYNAMIC_LIB_IS_INTERFACE)
    add_library(${CC_DYNAMIC_LIB_NAME} SHARED)
    string(REGEX REPLACE "^lib" "" LIB_NAME_WITHOUT_PREFIX ${CC_DYNAMIC_LIB_NAME})
    set_target_properties(${CC_DYNAMIC_LIB_NAME} PROPERTIES OUTPUT_NAME ${LIB_NAME_WITHOUT_PREFIX})

    # Set the output directory for the library
    if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
      set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/hami)
    endif()
    set_target_properties(${CC_DYNAMIC_LIB_NAME} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
      LINK_FLAGS "-Wl,--enable-new-dtags,-rpath,\\$ORIGIN"
    )
    message(STATUS "Dynamic library ${CC_DYNAMIC_LIB_NAME} will be built to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    

    target_sources(${CC_DYNAMIC_LIB_NAME} 
      PRIVATE ${CC_DYNAMIC_LIB_SRCS} ${CC_DYNAMIC_LIB_HDRS})
    
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

  else()
    message(FATAL_ERROR "Header only libraries are not supported by cc_dynamic_library. Consider using cc_library instead.")
  endif()

  # add alias for the library target
  add_library(:${CC_DYNAMIC_LIB_NAME} ALIAS ${CC_DYNAMIC_LIB_NAME})
endfunction()