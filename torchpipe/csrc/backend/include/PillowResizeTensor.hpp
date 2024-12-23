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
#include "cvcuda/PillowResizeCudaV2.hpp"

#include "Backend.hpp"
#include "dict.hpp"
namespace ipipe {
class Params;

/**
 * @brief PillowResize的gpu版本。
 * 修改自：
 *
 * https://github.com/CVCUDA/CV-CUDA/blob/release_v0.2.x/src/cvcuda/priv/legacy/pillow_resize.cu.
 */
class PillowResizeTensor : public SingleBackend {
 public:
  /**
   * @brief 设置相关参数。
   * @param resize_h 必需参数
   * @param resize_w 必需参数
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /**
   * @param TASK_DATA_KEY Tensor, 输入类型需要为kByte
   * @param[out] TASK_RESULT_KEY Tensor
   */
  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  unsigned resize_h_{0};
  unsigned resize_w_{0};
  std::unique_ptr<ipipe_nvcv::PillowResizeCudaV2> resizer_;
};

}  // namespace ipipe