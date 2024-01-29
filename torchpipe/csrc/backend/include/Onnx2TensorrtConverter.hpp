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

#include <ATen/ATen.h>

#include "Backend.hpp"
#include "dict.hpp"
#include "params.hpp"
#include "NvInferRuntime.h"

namespace ipipe {
class CudaEngineWithRuntime;
/**
 * @brief tensorrt模型转换器。接收的输入为onnx或者trt模型。
 *
 */
class Onnx2TensorrtConverter : public Backend {
 public:
  /**
   * @brief
   *
   * @param model 模型路径。以 .onnx 结尾的onnx文件，或者以 .trt 结尾的tensorrt
   * engine文件, 或者以 .onnx.encrypted 和 .trt.encrypted 结尾的加密文件。
   * @param min 输入为onnx时，网络最小形状。形式上， 可以是1, 1x3x224x224,
   * 多输入的模型则以逗号分开： 1,1； 字典序排序, 参见相关打印输出。
   * @param max 输入为onnx时，网络最大形状。形式上， 可以是4, 4x3x224x224,
   * 多输入的模型则以逗号分开： 4,4; 字典序排序，参见相关打印输出。
   * @param instance_num 实例数目；如果tensorrt
   engine的profile数目不足以建立足够的实例数，将反序列化多个engine.
   * @param precision
   网络要求的精度。模型为onnx时有效；如果精度不被显卡所支持，将自动退化到受支持的精度。默认
   fp32，支持fp32， fp16.
   @param mean 图像前处理中的减均值操作，会在onnx转tensorrt时，
   将该操作插入tensorrt网络中。 小于1时，会自动放大255倍。
   @param std 图像前处理中的除以方差的操作，会在onnx转tensorrt时，
   将该操作插入tensorrt网络中。 小于1时，会自动放大255倍。
   @param model::cache 自动缓存模型文件。支持 .trt 和 .trt.encrypted
   后缀的文件名。
   如果文件不存在，将自动保存该文件；否则将直接加载此文件。默认为空。
   EncryptHelper
   为加解密策略的默认参考实现，如有需要可以在源代码中修改相关策略，和密钥，以保证安全性。
   @param[out] _engine 模型解析成功将生成共享的engine。类型为
   `std::shared_ptr<CudaEngineWithRuntime>`.

   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  void forward(const std::vector<dict>& input_dicts) override final {
    throw std::runtime_error("forward should not be called");
  }

 private:
  std::unique_ptr<Params> params_;
  std::shared_ptr<CudaEngineWithRuntime> engine_;
  // std::unique_ptr<Backend> backend_;
  std::vector<std::vector<std::vector<int>>> mins_;
  std::vector<std::vector<std::vector<int>>> maxs_;  // profile - inputs - dim

  const std::set<std::string> valid_model_types_{
      ".trt",          ".onnx",        ".trt.encrypted",        ".onnx.encrypted",
      ".onnx.encrypt", ".trt.encrypt", ".trt.encrypted.buffer", ".onnx.encrypted.buffer",
      ".trt.buffer",   ".onnx.buffer"};
  bool is_valid_model_type(const std::string& model_type) const {
    bool valid = false;

    for (const auto& item : valid_model_types_) {
      valid = valid || endswith(model_type, item);
    }
    return valid;
  }
};

}  // namespace ipipe