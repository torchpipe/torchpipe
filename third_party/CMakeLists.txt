include(cc_library)
# include(cc_test)

cc_library(
  NAME 
    spdlog
  HDRS
    spdlog/spdlog.h
  INCLUDES
    spdlog/include
)

cc_library(
  NAME 
    toml
  INCLUDES
    tinytoml/
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