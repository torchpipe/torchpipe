include(cc_library)
include(cc_test)

cc_library(
  NAME 
    helper
  HDRS
    string.hpp
    base_logging.hpp
    macro.h
    hami_export.h
    params.hpp
    symbol.hpp
    threadsafe_sized_queue.hpp
  SRCS
    string.cpp
    base_logging.cpp
    params.cpp
    symbol.cpp
    toml_parser.cpp
    unique_index.cpp
    filesystem.cpp
  DEPS
    :spdlog
    :toml
    stdc++fs
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