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
 * @brief 删除键值，主要用于解决c++
 结果返回python端的时候，有些类别无法转换的问题。
 *
 *  * **使用示例**
    ```toml
    # 这里仅以toml配置文件方式展示RemoveV0的使用，其他方式使用同理：

    [out]
    backend = "RemoveV0"
    remove="color,model_struct"
    ```
 *

 */
class RemoveV0 : public SingleBackend {
 public:
  /**
   * @brief 初始化函数
   * @param remove 代表需要删除的键，多个键使用逗号分开。
   *
   * @remark "node_name", @ref TASK_DATA_KEY 和 @ref TASK_RESULT_KEY
   * 不能删除，所以不要将其放到remove参数中。
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config,
                    dict /*dict_config*/) {
    params_ = std::unique_ptr<Params>(new Params({{"remove", ""}, {}}, {}, {}, {}));

    if (!params_->init(config)) return false;
    auto strs = str_split(params_->at("remove"), ',');
    for (auto& key : strs) {
      if (startswith(key, "TASK_") && endswith(key, "_KEY")) {
        if (TASK_KEY_MAP.find(key) == TASK_KEY_MAP.end()) {
          SPDLOG_ERROR("not supportted: " + key);
          return false;
        }
        key = TASK_KEY_MAP.at(key);
        std::cout << key << std::endl;
      }
    }

    keys_ = std::set<std::string>(strs.begin(), strs.end());
    if (keys_.count(TASK_DATA_KEY) || keys_.count(TASK_RESULT_KEY) || keys_.count("node_name")) {
      SPDLOG_ERROR("TASK_DATA_KEY node_name or TASK_RESULT_KEY exists in configuration");
      return false;
    }

    return true;
  };

  void forward(dict input_dict) {
    assert(input_dict->find(TASK_DATA_KEY) != input_dict->end());
    (*input_dict)[TASK_RESULT_KEY] = (*input_dict)[TASK_DATA_KEY];
    if (!keys_.empty()) {
      for (auto iter = keys_.begin(); iter != keys_.end(); iter++) {
        std::string remove_key = *iter;
        auto map_iter = input_dict->find(remove_key);
        if (map_iter != input_dict->end()) {
          input_dict->erase(map_iter);
        } else {
          SPDLOG_WARN("Warning:[ RemoveV0 Backend ] : Not Fount This Key: [" + remove_key +
                      "] In Input Dict, Backend Skip RemoveV0 It.");
        }
      }
    }
  }

 private:
  std::set<std::string> keys_;
  std::unique_ptr<Params> params_;
};
IPIPE_REGISTER(Backend, RemoveV0, "RemoveV0");

class Add : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config,
                    dict /*dict_config*/) {
    params_ = std::unique_ptr<Params>(new Params({}, {"Add::backend"}, {}, {}));

    if (!params_->init(config)) return false;
    auto multiple_kv = str_split(params_->at("Add::backend"), ',');
    for (auto& single_kv : multiple_kv) {
      auto strs = str_split(single_kv, ':', true);
      IPIPE_ASSERT(strs.size() == 2);
      for (auto& key : strs) {
        try_replace_inner_key(key);
      }
      keys_[strs[0]] = strs[1];
    }

    return true;
  };

  void forward(dict input_dict) {
    for (const auto& item : keys_) {
      (*input_dict)[item.first] = item.second;
    }
  }

 private:
  std::unordered_map<std::string, std::string> keys_;
  std::unique_ptr<Params> params_;
};
IPIPE_REGISTER(Backend, Add, "Add,add");

class Copy : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config,
                    dict /*dict_config*/) {
    params_ = std::unique_ptr<Params>(new Params({}, {"Copy::backend"}, {}, {}));

    if (!params_->init(config)) return false;
    auto multiple_kv = str_split(params_->at("Copy::backend"), ',');
    for (auto& single_kv : multiple_kv) {
      auto strs = str_split(single_kv, ':');
      IPIPE_ASSERT(strs.size() == 2);
      for (auto& key : strs) {
        try_replace_inner_key(key);
      }
      keys_[strs[0]] = strs[1];
    }

    return true;
  };

  void forward(dict input_dict) {
    for (const auto& item : keys_) {
      auto iter = input_dict->find(item.first);
      IPIPE_ASSERT(iter != input_dict->end());
      (*input_dict)[item.first] = iter->second;
    }
  }

 private:
  std::unordered_map<std::string, std::string> keys_;
  std::unique_ptr<Params> params_;
};
IPIPE_REGISTER(Backend, Copy, "Copy,cp");

class Move : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config,
                    dict /*dict_config*/) {
    params_ = std::unique_ptr<Params>(new Params({}, {"Move::backend"}, {}, {}));

    if (!params_->init(config)) return false;
    auto multiple_kv = str_split(params_->at("Move::backend"), ',');
    for (auto& single_kv : multiple_kv) {
      auto strs = str_split(single_kv, ':');
      IPIPE_ASSERT(strs.size() == 2);
      for (auto& key : strs) {
        try_replace_inner_key(key);
      }
      keys_[strs[0]] = strs[1];
    }

    return true;
  };

  void forward(dict input_dict) {
    for (const auto& item : keys_) {
      auto iter = input_dict->find(item.first);
      IPIPE_ASSERT(iter != input_dict->end());
      (*input_dict)[item.first] = iter->second;
      input_dict->erase(iter);
    }
  }

 private:
  std::unordered_map<std::string, std::string> keys_;
  std::unique_ptr<Params> params_;
};
IPIPE_REGISTER(Backend, Move, "Move,mv");

class Remove : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config,
                    dict /*dict_config*/) {
    params_ = std::unique_ptr<Params>(new Params({}, {"Remove::backend"}, {}, {}));

    if (!params_->init(config)) return false;
    auto multiple_kv = str_split(params_->at("Remove::backend"), ',');
    for (const auto& single_kv : multiple_kv) {
      keys_.insert(single_kv);
    }

    return true;
  };

  void forward(dict input_dict) {
    for (const auto& item : keys_) {
      input_dict->erase(item);
    }
  }

 private:
  std::set<std::string> keys_;
  std::unique_ptr<Params> params_;
};
IPIPE_REGISTER(Backend, Remove, "Remove,Del,rm,del");
}  // namespace ipipe
