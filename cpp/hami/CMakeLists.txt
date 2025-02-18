add_subdirectory(core)
add_subdirectory(schedule)
add_subdirectory(helper)
add_subdirectory(builtin)

include(cc_dynamic_library)
include(cc_test)

cc_dynamic_library(
  NAME 
    libhami
  HDRS
    extension.hpp
  SRCS
    libhami.cpp
  DEPS
    :schedule
    :core
    :builtin
)

# LINKOPTS

# cc_test(
#   NAME 
#     common_test
#   SRCS
#     any_test.cpp
#   DEPS
#     common
#     GTest::gtest_main
# )

