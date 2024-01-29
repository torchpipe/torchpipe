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

#include <memory>

#include "Backend.hpp"
#include "dict.hpp"
namespace ipipe {
class Params;

/**
 * @brief 固定h，限制w最大值下，保持长宽比，进行resize和pad.
 */
class FixHResizePadMat : public Backend {
 public:
  /**
   * @param max_w 最大宽度值
   * @param resize_h 最大高度值
   * @param pad_value 常数pad值，默认为0，可设置一个或者三个,
   * 用逗号分开。
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /**
   * @param TASK_DATA_KEY cv::Mat. uint8 or
   * float，数据类型需要与pad_value一致。
   * @param TASK_RESULT_KEY cv::Mat
   */
  virtual void forward(const std::vector<dict>&) override;

 private:
  std::unique_ptr<Params> params_;
  int max_w_{0};
  int resize_h_{0};
  std::vector<float> pad_values_;
};

}  // namespace ipipe