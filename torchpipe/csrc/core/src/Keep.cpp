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

#include "params.hpp"

#include "Backend.hpp"
#include "dict.hpp"

#include "reflect.h"
#include "base_logging.hpp"
namespace ipipe {

/**
 * @brief 保留部分键值上的输出。主要用于解决 c++ ->
 * python 时有些类别无法转换的问题。
 */
class Keep : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config,
                    dict /*dict_config*/) {
    params_ = std::unique_ptr<Params>(new Params({{"keep", ""}, {}}, {}, {}, {}));

    if (!params_->init(config)) return false;
    auto strs = str_split(params_->at("keep"), ',');
    for (auto& key : strs) {
      if (startswith(key, "TASK_") && endswith(key, "_KEY")) {
        if (TASK_KEY_MAP.find(key) == TASK_KEY_MAP.end()) {
          SPDLOG_ERROR("not supportted: " + key);
          return false;
        }
        key = TASK_KEY_MAP.at(key);
      }
    }

    keys_ = std::set<std::string>(strs.begin(), strs.end());
    if (keys_.count(TASK_DATA_KEY) || keys_.count(TASK_RESULT_KEY)) {
      SPDLOG_ERROR("TASK_DATA_KEY or TASK_RESULT_KEY exists in configuration");
      return false;
    }

    return true;
  };

  void forward(dict input_dict) {
    assert(input_dict->find(TASK_DATA_KEY) != input_dict->end());
    (*input_dict)[TASK_RESULT_KEY] = (*input_dict)[TASK_DATA_KEY];
    if (!keys_.empty()) {
      for (auto iter = input_dict->begin(); iter != input_dict->end(); ++iter) {
        if (!keys_.count(iter->first)) {
          iter = input_dict->erase(iter);
        }
      }
    }
  }

 private:
  std::set<std::string> keys_;
  std::unique_ptr<Params> params_;
};
IPIPE_REGISTER(Backend, Keep, "Keep");
}  // namespace ipipe
