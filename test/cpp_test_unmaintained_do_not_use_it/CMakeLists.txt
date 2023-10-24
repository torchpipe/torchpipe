

# ---[ Test binaries.

file(GLOB_RECURSE IPIPE_ALL_TEST_FILES *.cpp)
message(status, "IPIPE_ALL_TEST_FILES=${IPIPE_ALL_TEST_FILES}")
if(BUILD_TEST)
  foreach(test_src ${IPIPE_ALL_TEST_FILES})
    get_filename_component(test_file_name ${test_src} NAME_WE)
    set(test_name "ipipe_${test_file_name}")
    add_executable(${test_name} "${test_src}")
    if(NOT MSVC)
      target_compile_options(${test_name} PRIVATE -Wno-unused-variable)
    endif()
    target_link_libraries(${test_name} ipipe gtest_main  gmock_main gmock gtest  pthread)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if(INSTALL_TEST)
      install(TARGETS ${test_name} DESTINATION test)
    endif()
  endforeach()
endif()