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
 * @brief jpg解码
 */
class DecodeMat : public SingleBackend {
 public:
  /**
   * @if chinese
   * @brief 解码。需要通道为3
   * @param TASK_DATA_KEY 类型为std::string， 待解码binary数据
   * @param[out] TASK_RESULT_KEY cv::Mat, int8, bgr, hwc
   * @param[out] color "bgr"
   * @else
   * @endif
   */
  virtual void forward(dict) override;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
class FixRatioResizeMat : public Backend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  // forward 按顺序调用， 不需要线程安全
  virtual void forward(const std::vector<dict>&) override;

 private:
  std::unique_ptr<Params> params_;
  int max_w_;
  int max_h_;
};
#endif

/**
 * @brief  已知resize_h, max_w，求得一批数据的目标尺寸[resize_h, w_i], 使得
 * w_i<=max_w， 且在此条件下最大限度保持长宽比。
 */
class BatchFixHLimitW : public Backend {
 public:
  /**
   * @param BatchFixHLimitW::max 指定max()返回值。
   * @param align 对齐，目标宽度需要是align的倍数。默认为32.
   * @param resize_h 目标高度。
   * @param max_w 最大宽度。
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /** @brief 送入至多BatchFixHLimitW::max个数据，求得各自的w，满足w<=max_w.
   * @param[out] max_w 目标宽度
   * @param[out] resize_h 目标高度。等于初始化时的值。
   * @param[out] TASK_RESULT_KEY input[TASK_RESULT_KEY] =
   * input[TASK_DATA_KEY];
   *
   */
  virtual void forward(const std::vector<dict>&) override;

  /**
   * @brief 由参数 BatchFixHLimitW::max 决定。
   */
  virtual uint32_t max() const override { return max_; };

 private:
  std::unique_ptr<Params> params_;
  uint32_t max_{1};
  int resize_h_;
  int align_;
  int max_w_{0};
};

/**
 * @brief 进行颜色空间转换（rgb<-->bgr）
 * @see cvtColorTensor
 */
class cvtColorMat : public Backend {
 public:
  /**
   * @param color 必须传入, 只能是 "rgb" 或者 "bgr"
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /**
   * @param color
   * 必须传入，如果与init时传入的值不同，将对输入数据进行颜色空间转换。
   * @param  TASK_DATA_KEY 类型为 cv::Mat
   */
  virtual void forward(const std::vector<dict>&) override;

 private:
  std::unique_ptr<Params> params_;
  std::string color_;
};

}  // namespace ipipe