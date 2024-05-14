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

#include "Backend.hpp"
#include "dict.hpp"
#include "params.hpp"

namespace ipipe {
/**
 * @brief 将 cv::Mat 变为 torch::Tensor(1chw);
 */
class Mat2Tensor : public SingleBackend {
 public:
  /**
   * @param device gpu(默认) 或者 cpu. 设置输出tensor的设备。参考 Mat2GpuTensor or Mat2CpuTensor
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /**
   * @param[in] TASK_DATA_KEY cv::Mat
   * @param[out] TASK_RESULT_KEY torch::Tensor， 1chw。
   */
  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  std::string data_format_;
};

/**
 * @brief Mat2Tensor with device = cpu;
 */
class Mat2CpuTensor : public Mat2Tensor {
 public:
  /**
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;
};

/**
 * @brief Mat2Tensor with device = gpu;
 */
class Mat2GpuTensor : public Mat2Tensor {
 public:
  /**
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
class Any2Mat : public SingleBackend {
 public:
  // forward 按顺序调用， 不需要线程安全
  virtual void forward(dict) override;
};
#endif

/**
 * @brief 将 torch::Tensor 变为 cv::Mat;
 */
class Tensor2Mat : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;
  /**
   * @param TASK_DATA_KEY torch::Tensor, 支持hwc 和 1hwc(其中 c==3)
   * @param[out] TASK_RESULT_KEY cv::Mat.
   */
  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
};
}  // namespace ipipe
