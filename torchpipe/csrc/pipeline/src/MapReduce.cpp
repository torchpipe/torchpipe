// Copyright 2021-2023 NetEase.
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

#include "MapReduce.hpp"
#include "base_logging.hpp"
#include "reflect.h"
#include "event.hpp"
#include "ipipe_common.hpp"
#include "exception.hpp"

namespace ipipe {

// 暂时先支持输入输出都为1个的子图跳转。
std::vector<dict> MapReduce::split(dict input) {
  if (split_.empty()) {
    return {input};
  }
  std::vector<dict> result;

  auto iter = input->find(split_);
  IPIPE_CHECK(iter != input->end(), "failed to find key: " + split_);

  auto split_size = iter->second.size();

  for (std::size_t i = 0; i < split_size; ++i) {
    auto out = make_dict("", input);
    TRACE_EXCEPTION((*out)[split_] = input->at(split_).at(i));

    result.push_back(out);
  }

  return result;
}

void MapReduce::merge(const std::vector<dict>& inputs, dict in_out) {
  for (const auto& item : merges_) {
    std::vector<any> data;
    for (auto input : inputs) {
      auto iter = input->find(item);
      if (iter == input->end()) {
        SPDLOG_ERROR("cann't find key " + item);
        throw std::invalid_argument("cann't find key " + item);
      } else
        data.push_back(iter->second);
    }
    (*in_out)[item] = data;
  }
  return;
}

bool MapReduce::post_init(const std::unordered_map<std::string, std::string>& config,
                          dict dict_config) {
  params_ =
      std::unique_ptr<Params>(new Params({{"split", "data"}, {"merge", "result"}}, {}, {}, {}));

  if (!params_->init(config)) return false;

  TRACE_EXCEPTION(split_ = params_->at("split"));
  TRACE_EXCEPTION(merges_ = str_split(params_->at("merge")));
  IPIPE_ASSERT(split_.find(',') == std::string::npos);
  std::for_each(merges_.begin(), merges_.end(),
                [](const std::string& data) { IPIPE_ASSERT(!data.empty()); });

  return !merges_.empty();
}

IPIPE_REGISTER(Backend, MapReduce, "MapReduce");

/**
 * @brief copy and split
 */
class CopySplit : public SingleBackend {
 public:
  /**
   */
  bool init(const std::unordered_map<std::string, std::string>& config,
            dict dict_config) override final {
    params_ = std::unique_ptr<Params>(new Params({{"split_size", "1"}}, {}, {}, {}));
    if (!params_->init(config)) return false;
    TRACE_EXCEPTION(split_size_ = std::stoi(params_->at("split_size")));
    IPIPE_ASSERT(split_size_ <= 1024 && split_size_ >= 1);
    return true;
  }
  /**
   * @brief
   * @param TASK_RESULT_KEY
   */
  virtual void forward(dict input_dict) override {
    auto iter = input_dict->find(TASK_DATA_KEY);
    auto data = std::vector<any>();
    for (std::size_t i = 0; i < split_size_; ++i) {
      data.push_back(iter->second);
    }

    (*input_dict)[TASK_RESULT_KEY] = data;
  }

 private:
  std::unique_ptr<Params> params_;
  int split_size_{0};
};

IPIPE_REGISTER(Backend, CopySplit, "CopySplit");

}  // namespace ipipe