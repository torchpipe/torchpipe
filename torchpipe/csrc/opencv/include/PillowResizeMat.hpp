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
namespace ipipe {
class Params;

/**
 * @brief 类似于 ResizeMat ，但结果与pillow的结果对齐。双线性插值。
 * 修改自 https://github.com/zurutech/pillow-resize
 * 性能上可能比不上opencv版本。
 */
class PillowResizeMat : public SingleBackend {
 public:
  /**
   * @brief 设置相关参数。
   * @param resize_h 必需参数
   * @param resize_w 必需参数
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /**
   * @param TASK_DATA_KEY cv::Mat, 数据类型不限，通道顺序支持 hwc, c==3.
   * @todo 支持的通道顺序与CropTensor对齐
   * @param[out] TASK_RESULT_KEY cv::Mat
   */
  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  unsigned resize_h_{0};
  unsigned resize_w_{0};
};

}  // namespace ipipe