include(cc_library)
include(cc_test)

cc_library(
  NAME
    schedule
  HDRS
    restart.hpp
  SRCS
    restart.cpp
    event_guard.cpp
    schedule.cpp
    contiguous_batching.cpp
    dag.cpp
    schedule_states.cpp
  DEPS
    :core
    :helper
    :thread_pool
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