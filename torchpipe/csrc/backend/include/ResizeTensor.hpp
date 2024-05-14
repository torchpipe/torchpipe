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

#include <torch/torch.h>

#include <ratio>

#include "Backend.hpp"
#include "dict.hpp"
#include "params.hpp"
namespace ipipe {
/**
 * @brief 对tensor进行直接的resize操作
 */
class ResizeTensor : public SingleBackend {
 public:
  /**
   * @brief 设置相关参数。
   * @param resize_h 必需参数
   * @param resize_w 必需参数
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /**
   * @param TASK_DATA_KEY torch::Tensor, 数据类型不限，通道顺序支持 1chw  和 hwc(deprecated) ,其中
   * c==1, 3, 4.
   * @param[out] TASK_RESULT_KEY torch::Tensor 1chw
   */
  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  unsigned resize_h_{0};
  unsigned resize_w_{0};
};

class ResizeTensorV1 : public SingleBackend {
 public:
  /**
   * @brief 设置相关参数。
   * @param resize_h 必需参数
   * @param resize_w 必需参数
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /**
   * @param TASK_DATA_KEY torch::Tensor, 数据类型不限，通道顺序支持 1chw  和 hwc(deprecated) ,其中
   * c==1, 3, 4.
   * @param[out] TASK_RESULT_KEY torch::Tensor 1chw
   */
  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  unsigned resize_h_{0};
  unsigned resize_w_{0};
};

}  // namespace ipipe
