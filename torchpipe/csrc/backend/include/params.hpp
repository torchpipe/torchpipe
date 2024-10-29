// Copyright 2021-2024 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
// #include "any.hpp"
#include "Backend.hpp"
#include "dict.hpp"
// #include "variant.hpp"
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
namespace ipipe {

// using config_value_type = std::variant()

/**
 * @brief 负责每个Backend的参数管理。这里主要维护4种类型的参数，分别为：
 * * init_optinal_params      初始化时的可选参数。
 * * init_required_params     初始化时的必要参数。
 * * forward_optinal_params   前向时的可选参数。
 * * forward_required_params  前向时的必要参数。
 *
 */
class Params {
 public:
  Params() {}

  Params(std::unordered_map<std::string, std::string> init_optinal_params,
         std::set<std::string> init_required_params = {},
         std::unordered_map<std::string, std::string> forward_optinal_params = {},
         std::set<std::string> forward_required_params = {},
         std::set<std::string> init_or_forward_required_params = {})
      : init_optinal_params_(init_optinal_params),
        init_required_params_(init_required_params),
        forward_optinal_params_(forward_optinal_params),
        forward_required_params_(forward_required_params) {}
  /**
   * @brief 初始化函数，主要作用在于：
   * * 将可选参数（init_optinal_params）保存到类内部参数_config中(
   * **内部参数_config会汇总所有需要的参数与对应值** )， 如果输入的参数config
   * 有更新值，将更新值保存到_config中。
   * * 判断config中是否有全部必须参数（init_required_params），并将其保存到_config中，如果没有会打印error，返回false，初始化失败。
   *
   * @param config 参数的键值对。
   * @return true or false
   */
  bool init(const std::unordered_map<std::string, std::string>& config);

  void check(const std::unordered_map<std::string, std::string>& config);

  /**
   * @brief 更新前向参数，主要作用在于：
   * * 将前向可选参数（forward_optinal_params）保存到_config中(
   * **内部参数_config会汇总所有需要的参数与对应值** )，
   * 如果输入参数config有更新值，将更新值保存到_config中。
   * * 判断config中是否有全部必须前向参数（formard_required_params），并将其保存到_config中，如果没有会打印error。
   *
   * @param forward_data 包含数据的键值对。
   */
  void check_and_update(dict forward_data);

  /**
   *  @brief 实现at取值。
   */
  std::string& at(const std::string& key) { return config_.at(key); }
  /**
   * @brief []重载，实现取值 。
   *
   */
  std::string& operator[](const std::string& key) { return config_[key]; }

  /**
   * @brief 将参数key-value插入到config_中。
   *
   */
  void set(const std::string& key, const std::string& value) { config_[key] = value; }

  /**
   * @brief 通过参数key，返回对应的值，如果没有，返回默认值。
   *
   */
  std::string get(const std::string& key, const std::string& default_value) {
    auto iter = config_.find(key);
    if (iter != config_.end()) {
      return iter->second;
    }
    return default_value;
  }

 private:
  std::unordered_map<std::string, std::string> config_;

  std::unordered_map<std::string, std::string> init_optinal_params_;
  std::set<std::string> init_required_params_;
  std::unordered_map<std::string, std::string> forward_optinal_params_;
  std::set<std::string> forward_required_params_;
  std::set<std::string> init_or_forward_required_params_;
};

struct MapInfo {
  enum MapType { replace, split, reduce };
  std::string value;

  enum MapType map_type = MapType::replace;
};

std::vector<std::string> str_split(std::string strtem, char a = ',',
                                   bool keep_empty_result = false);
std::vector<std::string> str_split_brackets_match(std::string strtem, char a, char left,
                                                  char right);

std::vector<float> strs2number(const std::string& data, char a = ',');
std::vector<int> str2int(const std::string& data, char in = ',');

template <typename T>
std::vector<T> str2number(const std::string& data, char in = ',') {
  if (data.empty()) return {};
  auto strs = str_split(data, in);
  std::vector<T> result;
  for (const auto& str_ : strs) {
    result.push_back(static_cast<T>(std::stod(str_)));
  }
  return result;
}

std::set<int> str2set(const std::string& data, char in = ',');
std::vector<std::set<int>> str2set(const std::string& data, char a, char b);

std::vector<std::vector<int>> str2int(const std::string& data, char a, char b);
std::unordered_map<std::string, std::unordered_map<std::string, MapInfo>> generate_map(
    std::string data);
mapmap generate_mapmap(std::string data);
std::set<std::string> get_map_reduce_nodes(const std::string& data);
static inline std::string join(const std::vector<int>& data, char in = ',') {
  std::stringstream ss;
  for (const auto& d : data) {
    ss << d << in;
  }
  std::string re = ss.str();
  if (!re.empty()) re.pop_back();

  return re;
}

/**
 * @brief 展开中括号：
 * 1. from A[B[C]] to  A; B; C ，
 * 2. from A[B[C],D,B[E[Z1,Z2]]] to A; B[C],D,B[E[Z1,Z2]]
 * 这里B[C],D,B[E[Z1,Z2]] 被当作一个整体。
 * @note A[B[C]]可自由添加空格
 */
std::vector<std::string> brackets_match(const std::string& strtem_, char left = '[',
                                        char right = ']');

/**
 * @brief 将中括号表达式转化为键值对参数形式；比如：
 *
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.py}
    # 从
    {"backend":"A[B[C]]"}
    # 展开为
    {"backend":"A","A::backend":"B","B::backend":"C"}
    # 从
    {"backend":"A[B[C],D,B[E[Z1,Z2]]]"}
    # 展开为
    {"backend":"A","A::backend":"B[C],D,B[E[Z1,Z2]]"}
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *    这里逗号运算符作为一个整体解析，`B[C],D,B[E[Z1,Z2]]` 作为A的后端
 * 的合法性依赖于A能够分别对三个后端`B[C]`，`D` 和 `B[E[Z1,Z2]]`
 * 中的括号进行理解， 也就是能对这三个后端分别调用 brackets_split
 * 函数展开为 {backend=B, B::backend=C} {backend=D} 和 {backend=B,
 * B::backend=E[Z1,Z2]}； 默认的Sequential 容器支持以上功能；
 *
 * 参见 @ref brackets_match.
 * @warning 不支持造成键值重复的情况；比如
 * B[B[C],D,B[E[Z1,Z2]]] 和 B[C[B]]是支持的，
 * 但是B[B[C]]不支持，因为后者会出现重复的B::backend 键值, 此时将抛出
 * invalid_argument 异常
 */
void brackets_split(const std::string& strtem, std::unordered_map<std::string, std::string>& config,
                    std::string key = "backend", char left = '[', char right = ']');

std::string pre_parentheses_split(const std::string& strtem, std::string& pre_str);
std::string post_parentheses_split(const std::string& strtem, std::string& post_str);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
void brackets_split(std::unordered_map<std::string, std::string>& config,
                    std::string key = "backend", char left = '[', char right = ']');
#endif

/// from {backend=A, A::backend=B,B::backend=C} to A[B[C]]
std::string brackets_combine(const std::unordered_map<std::string, std::string>& config);

inline bool endswith(std::string const& value, std::string const& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline std::string get_suffix(std::string input) {
  input = input.substr(input.find_last_of("/") + 1);
  std::string re;
  auto iter = input.find('.');
  if (iter != std::string::npos) {
    re = input.substr(iter);
  }
  return re;
}

inline std::string combine_strs(const std::set<std::string>& input, std::string delimiter = ",") {
  std::string re;
  if (input.empty()) return re;
  for (const auto& item : input) {
    re += item + delimiter;
  }
  re.substr(0, re.size() - 1);
  return re;
}

inline bool startswith(std::string const& value, std::string const& starting) {
  if (starting.size() > value.size()) return false;
  return std::equal(starting.begin(), starting.end(), value.begin());
}

void replace_all(std::string& s, std::string const& toReplace, std::string const& replaceWith);
std::string str_join(std::vector<std::string> datas, char sp);
std::string str_join(std::vector<std::string> datas);
void replace_all(std::string& s, std::string& original, std::string const& toReplace,
                 std::string const& replaceWith, const std::set<std::string>& valid_node,
                 std::string reduce_node_name, const std::set<std::string>& reduce_nodes);
void replace_task_key(std::string& key);

}  // namespace ipipe