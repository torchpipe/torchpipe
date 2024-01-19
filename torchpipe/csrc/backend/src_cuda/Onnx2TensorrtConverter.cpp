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

#include <fstream>
#ifdef WITH_TENSORRT
#include "Onnx2TensorrtConverter.hpp"
#include "base_logging.hpp"
#include "dynamic_onnx2trt.hpp"

#include "NativeAES.hpp"
#include "cuda_utils.hpp"
// #include "AES.h"

namespace ipipe {

bool Onnx2TensorrtConverter::init(const std::unordered_map<std::string, std::string>& config_param,
                                  dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"min", "1"},
                                                {"opt", "1"},
                                                {"max", "1"},
                                                {"instance_num", "1"},
                                                {"precision", ""},
                                                {"precision::fp32", ""},
                                                {"precision::fp16", ""},
                                                {"precision::output::fp32", ""},
                                                {"precision::output::fp16", ""},
                                                {"mean", ""},
                                                {"std", ""},
                                                {"model", ""},
                                                {"model::cache", ""},
                                                {"model::timingcache", ""},
                                                {"preprocessor", ""},
                                                {"model_type", ""},
                                                {"max_workspace_size", "1024"},
                                                {"allocator", "torch"},
                                                {"log_level", "info"}},
                                               {}, {}, {}));

  if (!params_->init(config_param)) return false;
  if (!dict_config) {
    SPDLOG_ERROR("dict_config == nullptr");
    return false;
  }

  auto& config = *dict_config;

  unsigned instance_num = std::stoi(params_->at("instance_num"));

  // std::vector<std::vector<std::vector<int>>> mins_;
  // std::vector<std::vector<std::vector<int>>> maxs_;
  if (!params_->at("min").empty() && params_->at("max").empty()) {
    params_->at("min") = params_->at("max");
  } else if (params_->at("min").empty() && !params_->at("max").empty()) {
    params_->at("max") = params_->at("min");
  } else if (params_->at("min").empty() && params_->at("max").empty()) {
    SPDLOG_ERROR("MultiTensorrtTensor: empty min and max shape");
    return false;
  }
  if (!params_->at("min").empty() && !params_->at("max").empty()) {
    auto min_shape = str_split(params_->at("min"), ';');
    auto max_shape = str_split(params_->at("max"), ';');

    for (const auto& item : min_shape) {
      mins_.push_back(str2int(item, 'x', ','));
      if (mins_.size() == instance_num) break;
    }
    for (const auto& item : max_shape) {
      maxs_.push_back(str2int(item, 'x', ','));
      if (maxs_.size() == instance_num) break;
    }

    if (min_shape.size() < instance_num) {
      mins_.resize(instance_num, mins_.back());
    }
    if (max_shape.size() < instance_num) {
      maxs_.resize(instance_num, maxs_.back());
    }
  }

  // 检查输入形状是否正确
  for (decltype(instance_num) profile_index = 0; profile_index < instance_num; ++profile_index) {
    bool error_shape = false;
    if (maxs_[profile_index].size() != mins_[profile_index].size()) {
      SPDLOG_ERROR(
          "number of inputs not match: maxs_[profile_index].size()({}) !=  "
          "mins_[profile_index].size()({})",
          maxs_[profile_index].size(), mins_[profile_index].size());
      return false;
    }
    for (std::size_t j = 0; j < maxs_[profile_index].size() && j < mins_[profile_index].size();
         ++j) {
      const auto local_max =
          std::min(maxs_[profile_index][j].size(), mins_[profile_index][j].size());
      for (std::size_t index_shape = 0; index_shape < local_max; ++index_shape) {
        if (maxs_[profile_index][j][index_shape] < mins_[profile_index][j][index_shape]) {
          error_shape = true;
        }
      }
    }
    // 形状有误， 则打印出来
    if (error_shape) {
      std::stringstream ss;
      ss << "mins_ & maxs_ not match; mins_ =";
      for (std::size_t j = 0; j < mins_[profile_index].size() && j < maxs_[profile_index].size();
           ++j) {
        for (std::size_t index_shape = 0; index_shape < mins_[profile_index][j].size();
             ++index_shape) {
          ss << " " << mins_[profile_index][j][index_shape];
        }
        // ss << "\n";
      }
      ss << "; maxs_ = ";
      for (std::size_t j = 0; j < mins_[profile_index].size() && j < maxs_[profile_index].size();
           ++j) {
        for (std::size_t index_shape = 0; index_shape < maxs_[profile_index][j].size();
             ++index_shape) {
          if (0 != index_shape)
            ss << "x" << maxs_[profile_index][j][index_shape];
          else
            ss << maxs_[profile_index][j][index_shape];
        }
        // ss << "\n";
      }
      ss << ";";

      SPDLOG_ERROR(ss.str());
      return false;
    }
  }

  std::string model = params_->at("model");

  if (endswith(model, ".trt")) {
    if (!params_->at("mean").empty() || !params_->at("std").empty()) {
      SPDLOG_ERROR("mean and std  are not supported by .trt model.");
      return false;
    }
  }

  std::vector<std::string> precision_fp32;
  std::vector<std::string> precision_fp16;
  std::vector<std::string> precision_output_fp32;
  std::vector<std::string> precision_output_fp16;
  if (!params_->at("precision::fp32").empty()) {
    precision_fp32 = str_split(params_->at("precision::fp32"));
    IPIPE_ASSERT(!endswith(model, ".trt"));
    SPDLOG_INFO("these layers keep fp32: {}", params_->at("precision::fp32"));
  }
  if (!params_->at("precision::fp16").empty()) {
    precision_fp16 = str_split(params_->at("precision::fp16"));
    SPDLOG_INFO("these layers keep fp16: {}", params_->at("precision::fp16"));
    IPIPE_ASSERT(!endswith(model, ".trt"));
  }
  if (!params_->at("precision::output::fp32").empty()) {
    precision_output_fp32 = str_split(params_->at("precision::output::fp32"));
    IPIPE_ASSERT(!endswith(model, ".trt"));
    SPDLOG_INFO("these layers' outputs are fp32: {}", params_->at("precision::output::fp32"));
  }
  if (!params_->at("precision::output::fp16").empty()) {
    precision_output_fp16 = str_split(params_->at("precision::fp16"));
    SPDLOG_INFO("these layers' outputs are keep fp16: {}", params_->at("precision::output::fp16"));
    IPIPE_ASSERT(!endswith(model, ".trt"));
  }

  bool need_cache = false;
  auto Is_File_Exist = [](const std::string& file_path) {
    std::ifstream file(file_path.c_str());
    return file.good();
  };

  if (!params_->at("model::cache").empty()) {
    if (Is_File_Exist(params_->at("model::cache"))) {
      SPDLOG_WARN("Using cached model [{}]. {}.", params_->at("model::cache"),
                  "Delete it to regenerate.\n");
      model = params_->at("model::cache");
    } else {
      need_cache = true;
      IPIPE_ASSERT(!model.empty());
    }
  }

  std::string model_type = params_->at("model_type");
  if (model_type.empty()) model_type = get_suffix(model);
  if (!is_valid_model_type(model_type)) {
    SPDLOG_ERROR("invalid suffix. Support: \n" + combine_strs(valid_model_types_, "\n"));
    return false;
  }
  if (endswith(model_type, ".encrypted") || endswith(model_type, ".encrypted")) {
    model = decrypt_data(model_type, model);
  }

  if (!is_valid_model_type(model_type)) {
    SPDLOG_ERROR("invalid suffix. given {}, Support: \n" + combine_strs(valid_model_types_, "\n"),
                 model_type);
    return false;
  }

  if (!endswith(model_type, ".buffer") && model_type.size() > 260) {
    SPDLOG_ERROR("sizeof file name > 260");
    return false;
  }

  if (!initPlugins()) {
    SPDLOG_WARN("initLibNvInferPlugins failed");
  }
  std::string engine_plan;
  if (endswith(model_type, ".trt") || endswith(model_type, ".trt.encrypt")) {
    engine_ = loadCudaBackend(model, model_type, engine_plan);
  } else if (endswith(model_type, ".trt.buffer")) {
    engine_plan = model;

    engine_ = loadEngineFromBuffer(engine_plan);

  } else if (endswith(model_type, ".onnx") || endswith(model_type, ".onnx.buffer") ||
             endswith(model_type, ".onnx.encrypt")) {
    std::vector<float> means = strs2number(params_->at("mean"));
    std::vector<float> stds = strs2number(params_->at("std"));

    OnnxParams onnxp;
    auto max_workspace_size = std::stoi(params_->at("max_workspace_size"));
    onnxp.max_workspace_size = 1024 * 1024 * max_workspace_size;
    IPIPE_ASSERT(max_workspace_size >= 1);
    onnxp.timecache = params_->at("model::timingcache");
    onnxp.precision = params_->at("precision");
    onnxp.allocator = params_->at("allocator");

    if (onnxp.precision.empty()) {
      auto sm = get_sm();
      if (sm <= "6.1")
        onnxp.precision = "fp32";
      else
        onnxp.precision = "fp16";
      SPDLOG_WARN(
          "'precision' not set. You can set it to one of [fp16|fp32|int8|best]. Default to fp16 if "
          "platformHasFastFp16 and SM>6.1 else fp32.\n");
    }
    onnxp.precision_fp32 = std::set<std::string>(precision_fp32.begin(), precision_fp32.end());
    onnxp.precision_fp16 = std::set<std::string>(precision_fp16.begin(), precision_fp16.end());
    onnxp.precision_output_fp32 =
        std::set<std::string>(precision_output_fp32.begin(), precision_output_fp32.end());
    onnxp.precision_output_fp16 =
        std::set<std::string>(precision_output_fp16.begin(), precision_output_fp16.end());
    onnxp.log_level = params_->at("log_level");

    engine_ =
        onnx2trt(model, model_type, mins_, maxs_, engine_plan, onnxp, config_param, means, stds);
  } else {
    throw std::runtime_error("invalid model type: " + model_type);
  }
  if (!engine_) {
    return false;
  }
  if (need_cache) {
    if (endswith(params_->at("model::cache"), ".trt")) {
      std::ofstream ff(params_->at("model::cache"));
      ff << engine_plan;
      SPDLOG_INFO("model cached: {}, size = {}MB", params_->at("model::cache"),
                  int(100 * engine_plan.size() / 1024.0 / 1024.0) / 100.0);
    } else if (endswith(params_->at("model::cache"), ".trt.encrypted")) {
      encrypt_buffer_to_file(engine_plan, params_->at("model::cache"), "");
    } else {
      SPDLOG_ERROR("model::cache should end with .trt or .trt.encrypted");
      return false;
    }
  }

  config["_engine"] = engine_;
  return true;
}
}  // namespace ipipe

#endif