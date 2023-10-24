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
 * @brief cv::Mat 的数据类型转为float。
 * @see ResizeTensor
 */
class FloatMat : public SingleBackend {
 public:
  /**
   * @param TASK_DATA_KEY cv::Mat, 数据类型不限，通道顺序支持 hwc, c==3.
   * @param[out] TASK_RESULT_KEY cv::Mat
   */
  virtual void forward(dict) override;

 private:
};

}  // namespace ipipe