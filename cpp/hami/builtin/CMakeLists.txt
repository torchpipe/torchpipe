include(cc_library)
include(cc_test)

cc_library(
  NAME 
    builtin
  HDRS
    identity.hpp
    aspect.hpp
    basic_backends.hpp
    proxy.hpp 
    page_table.hpp
  SRCS
    identity.cpp
    sequential.cpp
    condition.cpp
    select.cpp
    launch.cpp 
    result_parser.cpp
    aspect.cpp
    basic_backends.cpp
    register_node.cpp
    interpreter.cpp
    proxy.cpp
    cat_split.cpp
    ioc.cpp
    generate_backend.cpp
    result_queue.cpp
    benchmark.cpp
    key_operator.cpp
    source.cpp
    control_plane.cpp
    page_table.cpp
    log.cpp
  DEPS
    :spdlog
    :core
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