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

#ifdef WITH_TORCH
#include <opencv2/core.hpp>
#include "Mat2Tensor.hpp"
#include <thread>
#include "base_logging.hpp"
#include "reflect.h"
#include "torch_mat_utils.hpp"
#include "Backend.hpp"
#include "torch_utils.hpp"
#include "exception.hpp"
namespace ipipe {

bool Mat2Tensor::init(const std::unordered_map<std::string, std::string>& config_param,
                      dict dict_config) {
  params_ = std::unique_ptr<Params>(
      new Params({{"device", "gpu"}, {"data_format", "nchw"}, {"node_name", ""}}, {}, {}, {}));
  if (!params_->init(config_param)) return false;
  TRACE_EXCEPTION(data_format_ = params_->at("data_format"));
  IPIPE_ASSERT(data_format_ == "nchw" || data_format_ == "hwc");
  SPDLOG_INFO("Mat2Tensor: device = {}. You can also use Mat2GpuTensor or Mat2CpuTensor.",
              params_->at("device"));
  if (torch_is_using_default_stream()) {
    SPDLOG_WARN(
        "Mat2Tensor runs in default stream. This is not an error but it will affect the "
        "performance. Inserting SyncTensr by  "
        "S[...,Mat2Tensor,SyncTensor] or SyncTensor[Mat2Tensor] if you didn't do it on purpose.\n");
  }

  return true;
}

void Mat2Tensor::forward(dict input_dict) {
  auto& input = *input_dict;

  auto iter = input_dict->find(TASK_DATA_KEY);
  if (iter != input_dict->end()) {
    if (iter->second.type() == typeid(cv::Mat)) {
      cv::Mat data = any_cast<cv::Mat>(iter->second);
      if (params_->at("device") == "cpu") {
        input[TASK_RESULT_KEY] = cvMat2TorchCPU(data, true, data_format_);
        input["data_format"] = data_format_;
      } else {
        input["data_format"] = data_format_;
        input[TASK_RESULT_KEY] = cvMat2TorchGPU(data, data_format_);
      }

    } else if (iter->second.type() == typeid(std::vector<cv::Mat>)) {
      const std::vector<cv::Mat>& data = any_cast<std::vector<cv::Mat>>(iter->second);
      std::vector<at::Tensor> result;
      for (const auto& d : data) {
        if (params_->at("device") == "cpu") {
          result.emplace_back(cvMat2TorchCPU(d, true, data_format_));
        } else
          result.emplace_back(cvMat2TorchGPU(d, data_format_));
      }

      input[TASK_RESULT_KEY] = result;
      input["data_format"] = data_format_;
    } else {
      SPDLOG_ERROR("unknown type: {}", iter->second.type().name());
      throw std::runtime_error("unknown type: " + std::string(iter->second.type().name()));
    }
  } else {
    SPDLOG_ERROR("not found: {}", TASK_DATA_KEY);
    throw std::runtime_error("data not found");
  }
}

bool Mat2CpuTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                         dict dict_config) {
  auto new_config_param = config_param;
  new_config_param["device"] = "cpu";
  return Mat2Tensor::init(new_config_param, dict_config);
}

bool Mat2GpuTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                         dict dict_config) {
  auto new_config_param = config_param;
  new_config_param["device"] = "gpu";
  return Mat2Tensor::init(new_config_param, dict_config);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
void Any2Mat::forward(dict input_dict) {
  auto& input = *input_dict;

  auto iter = input_dict->find(TASK_DATA_KEY);
  if (iter != input_dict->end()) {
    if (iter->second.type() == typeid(std::vector<any>)) {
      std::vector<any> data = any_cast<std::vector<any>>(iter->second);
      if (data.empty()) {
        input[TASK_RESULT_KEY] = std::vector<cv::Mat>();
      } else {
        std::vector<cv::Mat> mats;
        for (auto& item : data) {
          cv::Mat da = any_cast<cv::Mat>(item);
          mats.push_back(da);
        }
        input[TASK_RESULT_KEY] = mats;
      }
    }
  }
}
IPIPE_REGISTER(Backend, Any2Mat, "Any2Mat");
#endif

IPIPE_REGISTER(Backend, Mat2Tensor, "Mat2Tensor");
IPIPE_REGISTER(Backend, Mat2CpuTensor, "Mat2CpuTensor");
IPIPE_REGISTER(Backend, Mat2GpuTensor, "Mat2GpuTensor");

bool Tensor2Mat::init(const std::unordered_map<std::string, std::string>& config_param,
                      dict dict_config) {
  if (torch_is_using_default_stream()) {
    SPDLOG_WARN(
        "Tensor2Mat runs in default stream. This is not an error but it will affect the "
        "performance。 Inserting SyncTensr by  "
        "S[Tensor2Mat,SyncTensor] or SyncTensor[Tensor2Mat] if you didn't do it on purpose.\n");
  }
  return true;
}

void Tensor2Mat::forward(dict input_dict) {
  auto& input = *input_dict;

  auto data = dict_get<at::Tensor>(input_dict, TASK_DATA_KEY);
  auto result = torchTensortoCVMatV2(data, true);  // true is for 'deepcopy'
  input[TASK_RESULT_KEY] = result;
}

IPIPE_REGISTER(Backend, Tensor2Mat, "Tensor2Mat");

}  // namespace ipipe
#endif