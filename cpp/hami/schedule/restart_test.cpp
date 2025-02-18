// restart_test.cpp

#include <gtest/gtest.h>
#include "hami/core/restart.hpp"
#include "hami/core/reflect.h"
#include <unordered_map>
#include <string>
#include <any>

namespace hami {

// 模拟 dict 类型
using dict = std::unordered_map<std::string, std::any>;

class RestartTest : public ::testing::Test {
 protected:
  std::unique_ptr<Backend> restart;

  void SetUp() override {
    // 使用HAMI_CREATE创建Restart实例
    restart = std::unique_ptr<Backend>(HAMI_CREATE(Backend, "Restart"));
    ASSERT_NE(restart, nullptr);
  }
};

TEST_F(RestartTest, InitializationTest) {
  std::unordered_map<std::string, std::string> config;
  dict dict_config;

  // 测试初始化
  EXPECT_NO_THROW(restart->init(config, dict_config));
}

TEST_F(RestartTest, ForwardTest) {
  std::unordered_map<std::string, std::string> config;
  dict dict_config;
  restart->init(config, dict_config);

  std::vector<dict> inputs = {{{"key1", "value1"}}, {{"key2", "value2"}}};

  // 测试forward方法
  EXPECT_NO_THROW(restart->forward(inputs));
}

TEST_F(RestartTest, MinMaxTest) {
  std::unordered_map<std::string, std::string> config;
  dict dict_config;
  restart->init(config, dict_config);

  // 测试min和max方法
  EXPECT_GT(restart->min(), 0);
  EXPECT_GT(restart->max(), 0);
}

// 添加更多测试用例来覆盖Restart类的其他功能

}  // namespace hami
