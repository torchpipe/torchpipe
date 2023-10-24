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

#pragma once

#include <memory>

#include "Backend.hpp"
#include "dict.hpp"
#include "params.hpp"

namespace ipipe {

/**
 * @brief 限制hw最大值下，保持长宽比，进行resize和pad.
 */
class ResizePad : public SingleBackend {
 public:
  /**
   * @param max_w 最大宽度值
   * @param max_h 最大高度值
   * @param pad_value 常数pad值，默认为0.
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param, dict) {
    params_ = std::unique_ptr<Params>(new Params({{"pad_value", "0"}}, {"max_w", "max_h"}, {}, {}));
    if (!params_->init(config_param)) return false;

    max_w_ = std::stoi(params_->at("max_w"));
    max_h_ = std::stoi(params_->at("max_h"));
    IPIPE_ASSERT(max_h_ >= 1 && max_h_ <= 1024 * 1024 && max_w_ >= 1 && max_w_ <= 1024 * 1024);
    auto pads = str_split(params_->at("pad_value"));
    for (const auto& item : pads) pad_values_.push_back(std::stof(item));

    if (pad_values_.empty()) return false;
    while (pad_values_.size() < 3) {
      pad_values_.push_back(pad_values_.back());
    }

    return true;
  }

  /**
   * @param[in] TASK_DATA_KEY  uint8 or
   * float，数据类型需要与pad_value一致。
   * @param[out] TASK_RESULT_KEY 输出坐标
   * @param[out] inverse_trans std::function<std::pair<float, float>(float x,
   * float y)>. 用于将新坐标映射到原始坐标。
   */
  virtual void forward(dict) = 0;

 protected:
  std::unique_ptr<Params> params_;
  int max_w_;
  int max_h_;
  std::vector<float> pad_values_;
};

}  // namespace ipipe