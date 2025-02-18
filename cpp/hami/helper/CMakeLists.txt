include(cc_library)
include(cc_test)

cc_library(
  NAME 
    helper
  HDRS
    string.hpp
    base_logging.hpp
    macro.h
    params.hpp
    symbol.hpp
    threadsafe_sized_queue.hpp
  SRCS
    string.cpp
    base_logging.cpp
    params.cpp
    symbol.cpp
  DEPS
    :spdlog
    :toml
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