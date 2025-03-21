
#include <algorithm>
#include <cmath>

#include "hami/builtin/basic_backends.hpp"
#include "hami/core/helper.hpp"
#include "hami/core/reflect.h"
#include "hami/core/task_keys.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/macro.h"
#include "hami/helper/string.hpp"

namespace hami {

class Add : public BackendOne {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override {
    parser_v2::ArgsKwargs args_kwargs =
        parser_v2::get_args_kwargs(this, "Add", params);
    HAMI_ASSERT(
        args_kwargs.first.size() >= 1,
        "Requires exactly >=1 argument. Usage: Add(a:b,c:d)/Add::args=a:b,c:d");
    for (const auto& arg : args_kwargs.first) {
      SPDLOG_INFO("Add  " + arg);
      auto keys = str::str_split(arg, ':');
      HAMI_ASSERT(keys.size() == 2, "Invalid argument: " + arg);
      keys_[keys[0]] = keys[1];
    }

    try_replace_inner_key(keys_);
    has_result_ = keys_.find(TASK_RESULT_KEY) != keys_.end();
  }

  void forward(const dict& input_dict) override {
    for (const auto& item : keys_) {
      (*input_dict)[item.first] = item.second;
    }
    if (!has_result_)
      (*input_dict)[TASK_RESULT_KEY] = input_dict->at(TASK_DATA_KEY);
  }

 private:
  std::unordered_map<std::string, std::string> keys_;
  bool has_result_{false};
};
HAMI_REGISTER(Backend, Add);

class AddInt : public BackendOne {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& kwargs) override {
    parser_v2::ArgsKwargs args_kwargs =
        parser_v2::get_args_kwargs(this, "Add", params);
    HAMI_ASSERT(
        args_kwargs.first.size() >= 1,
        "Requires exactly >=1 argument. Usage: AddInt(a:b,c:d)/AddInt::args=a:b,c:d");
    for (const auto& arg : args_kwargs.first) {
      SPDLOG_INFO("Add  " + arg);
      auto keys = str::str_split(arg, ':');
      HAMI_ASSERT(keys.size() == 2, "Invalid argument: " + arg);
      keys_[keys[0]] = std::stoi(keys[1]);
    }

    try_replace_inner_key(keys_);
    has_result_ = keys_.find(TASK_RESULT_KEY) != keys_.end();
  }
  void forward(const dict& input_dict) override {
    for (const auto& item : keys_) {
      (*input_dict)[item.first] = item.second;
    }
    if (!has_result_)
      (*input_dict)[TASK_RESULT_KEY] = input_dict->at(TASK_DATA_KEY);
  }

 private:
  std::unordered_map<std::string, int> keys_;
  bool has_result_{false};
};
HAMI_REGISTER(Backend, AddInt);

class Remove : public BackendOne {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override {
    std::string dep = get_dependency_name_force(this, config);
    auto keys = str::str_split(dep, ',');
    for (auto key : keys_) {
      try_replace_inner_key(key);
      keys_.insert(key);
    }

    has_result_ = keys_.find(TASK_RESULT_KEY) != keys_.end();
  }
  void forward(const dict& input_dict) {
    for (const auto& item : keys_) {
      input_dict->erase(item);
    }
    if (!has_result_)
      (*input_dict)[TASK_RESULT_KEY] = input_dict->at(TASK_DATA_KEY);
  }

 private:
  std::unordered_set<std::string> keys_;
  bool has_result_{false};
};
HAMI_REGISTER(Backend, Remove);

} // namespace hami