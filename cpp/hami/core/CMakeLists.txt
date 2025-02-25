include(cc_library)
include(cc_test)

cc_library(
  NAME 
    core
  HDRS
    any.hpp
    dict.hpp
    event.hpp
    reflect.h
    backend.hpp
    helper.hpp
    task_keys.hpp
  SRCS
    event.cpp
    reflect.cpp
    backend.cpp
    helper.cpp
  DEPS
    :helper
)

# cc_test(
#   NAME 
#     common_test
#   SRCS
#     any_test.cpp
#   DEPS
#     common
#     GTest::gtest_main
# )