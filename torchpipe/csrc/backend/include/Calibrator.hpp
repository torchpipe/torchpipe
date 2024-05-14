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
#include "prepost.hpp"
#include <torch/torch.h>

#include <NvInfer.h>
#include <string>
#include <vector>

namespace ipipe {

class Calibrator : public Backend, public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;
  int getBatchSize() const noexcept override;
  bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
  const void* readCalibrationCache(size_t& length) noexcept override;
  void writeCalibrationCache(const void* cache, size_t length) noexcept override;
  // virtual uint32_t max() override { return 1; }

  void forward(const std::vector<dict>& input_dicts) override final {
    throw std::runtime_error("forward should not be called");
  }

 private:
  std::unique_ptr<Params> params_;
  std::unique_ptr<PreProcessor<torch::Tensor>> preprocessor_;
  int batchsize_;
  std::string calib_table_name_;
  //   std::string calibration_tensor_dir_;
  std::size_t read_batch_index_ = 0;
  std::vector<std::string> files_;
  std::vector<char> calib_cache_;
  int max_batch_num_;
};
}  // namespace ipipe