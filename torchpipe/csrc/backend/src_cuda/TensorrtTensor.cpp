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

#ifdef WITH_TENSORRT
#include <ATen/ATen.h>
#include <nppcore.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <thread>
// #include "DynamicOnnx2TrtBackend.hpp"
#include "base_logging.hpp"
#include "Backend.hpp"
#include "dict.hpp"
#include "c10/cuda/CUDAStream.h"
#include "dynamic_onnx2trt.hpp"
#include "hw_batching.hpp"

#include "prepost.hpp"
#include "reflect.h"
#include "time_utils.hpp"
#include "torch_utils.hpp"
#include "base_logging.hpp"
#include "exception.hpp"
#include "tensorrt_utils.hpp"
#include "TensorrtTensor.hpp"
#include "Onnx2TensorrtConverter.hpp"
#include "SingleConcatPreprocess.hpp"
#include "MultipleConcatPreprocess.hpp"
// https://github.com/NVIDIA/Torch-TensorRT/blob/3a98a8b198a071e622c43283caea7416fe8a8a1a/core/runtime/register_trt_op.cpp

// #define USE_OUT_MEM
namespace ipipe {

bool TensorrtTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                          dict dict_config) {
  if (torch_is_using_default_stream()) {
    SPDLOG_WARN(
        "TensorrtTensor runs in default stream. This would cause error if downstream nodes use "
        "different CUDA streams. Inserting SyncTensr by  "
        "S[...,TensorrtTensor,SyncTensor] or SyncTensor[TensorrtTensor] if you didn't do it on "
        "purpose.\n");
  }
  params_ = std::unique_ptr<Params>(new Params({{"postprocessor", "split"},
                                                {"preprocessor", ""},
                                                {"_independent_thread_index", "0"},
                                                {"TensorrtTensor::backend", ""},
                                                {"save_engine", ""},
                                                {"instance_num", "1"},
                                                {"force_range", ""}},
                                               {}, {}, {}));

#if NV_TENSORRT_MAJOR < 7
  SPDLOG_ERROR("tensorrt version should >= 7.2 but got {}.{}", NV_TENSORRT_MAJOR,
               NV_TENSORRT_MINOR);
  return false;

#elif NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 2
  SPDLOG_WARN("tensorrt version should >= 7.2 but got {}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);

#endif

  if (!params_->init(config_param)) return false;
  if (!dict_config) {
    dict_config = std::make_shared<std::unordered_map<std::string, any>>();
  }
  // get engine
  auto& config = *dict_config;
  if (config.count("_engine") == 0) {
    assert(!backend_);
    if (params_->at("TensorrtTensor::backend").empty()) {
      backend_ = std::unique_ptr<Backend>(new Onnx2TensorrtConverter());
    } else {
      backend_ =
          std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("TensorrtTensor::backend")));
    }
    if (!backend_ || !backend_->init(config_param, dict_config)) {
      return false;
    }
  }

  backend_.reset();  // release memory
  assert(config.count("_engine") != 0);

  engine_ = any_cast<std::shared_ptr<CudaEngineWithRuntime>>(config.at("_engine"));
  IPIPE_ASSERT(engine_ && engine_->engine);

  unique_ptr_destroy<nvinfer1::IHostMemory> p_engine_plan{engine_->engine->serialize()};

  std::string engine_plan = std::string((char*)p_engine_plan->data(), p_engine_plan->size());

  int _independent_thread_index = 0;
  // set tensorrt profile
  if (!params_->at("_independent_thread_index").empty()) {
    TRACE_EXCEPTION(_independent_thread_index =
                        std::stoi(params_->at("_independent_thread_index")));
  } else {
    _independent_thread_index = 0;
  }

  independent_thread_index_ = _independent_thread_index;
  int instance_num = std::stoi(params_->at("instance_num"));
  if (instance_num <= _independent_thread_index) {
    SPDLOG_ERROR("instance_num <= _independent_thread_index: " + std::to_string(instance_num) +
                 " <= " + std::to_string(_independent_thread_index));
    return false;
  }

  if (engine_->engine->getNbOptimizationProfiles() < instance_num &&
      _independent_thread_index == 0) {
    SPDLOG_INFO(
        "Number of OptimizationProfiles({}) < instance_num({}). The engine will be repeatly "
        "used.\n",
        engine_->engine->getNbOptimizationProfiles(), instance_num);
  }

  parse_context(dict_config, _independent_thread_index);

  change_shape_ = std::vector<bool>(maxs_.size(), true);

  /********post*****/
  std::string batch_post = params_->at("postprocessor");

  postprocessor_ = std::unique_ptr<PostProcessor<at::Tensor>>(
      IPIPE_CREATE(PostProcessor<at::Tensor>, batch_post));
  try {
    if (!postprocessor_ || !postprocessor_->init(config_param, dict_config)) {
      SPDLOG_ERROR("error postprocessor: " + batch_post);
      return false;
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("error postprocessor({}): {}", batch_post, e.what());
    return false;
  }

  /********pre*****/
  std::string preprocessor = params_->at("preprocessor");

  if (preprocessor.empty()) {
    if (mins_.size() == 1) {
      preprocessor_ = std::unique_ptr<PreProcessor<at::Tensor>>(new SingleConcatPreprocess());
    } else {
      preprocessor_ = std::unique_ptr<PreProcessor<at::Tensor>>(new MultipleConcatPreprocess());
    }
  } else {
    preprocessor_ = std::unique_ptr<PreProcessor<at::Tensor>>(
        IPIPE_CREATE(PreProcessor<at::Tensor>, preprocessor));
  }

  (*dict_config)["max"] = maxs_;
  (*dict_config)["min"] = mins_;

  if (!preprocessor_ || !preprocessor_->init(config_param, dict_config)) {
    SPDLOG_ERROR("preprocess_engine created({}) or inited failed .", bool(preprocessor_));
    return false;
  }

  // save engne if needed.
  // if (!params_->at("save_engine").empty() && _independent_thread_index == 0) {
  //   SPDLOG_INFO("engine saved:  {}{}", params_->at("save_engine"), "\n");
  //   std::ofstream ff(params_->at("save_engine"));
  //   ff << engine_plan;
  // }

  if (maxs_.empty() || maxs_[0].empty() || mins_.empty() || mins_[0].empty()) {
    SPDLOG_ERROR("maxs_.empty() || maxs_[0].empty() || mins_.empty() || mins_[0].empty()");
    return false;
  }

  std::vector<std::vector<int>> force_ranges;
  TRACE_EXCEPTION(force_ranges = str2int(params_->at("force_range"), ',', ';'));
  max_ = maxs_[0][0];
  min_ = mins_[0][0];

  if (!force_ranges.empty()) {
    while (force_ranges.size() < instance_num) {
      force_ranges.push_back(force_ranges.back());
    }
    IPIPE_ASSERT(force_ranges.size() == instance_num);

    const auto& force_range = force_ranges[independent_thread_index_];
    IPIPE_ASSERT(force_range.size() == 2);
    max_ = force_range[1];
    min_ = force_range[0];
    SPDLOG_DEBUG("force_range: thread_index={} min = {}, max= {}", independent_thread_index_, min_,
                 max_);
  }
  IPIPE_ASSERT(min_ >= 1 && max_ >= min_);

  return true;
}

void TensorrtTensor::parse_context(dict dict_config, int _independent_thread_index) {
  auto& config = *dict_config;
  // 获取网络输入输出信息
  const int n_profiles = engine_->engine->getNbOptimizationProfiles();

  profile_index_ = _independent_thread_index % n_profiles;
  // engine 没有多余的profile， 下一个实例重新初始化
  if (profile_index_ == n_profiles - 1) {
    config.erase("_engine");
    // config.erase("_engine_raw");
  }

#ifdef USE_OUT_MEM
  context_ = unique_ptr_destroy<nvinfer1::IExecutionContext>(
      engine_->engine->createExecutionContextWithoutDeviceMemory());
  auto mem = torch_allocate(engine_->engine->getDeviceMemorySize());
  context_->setDeviceMemory(mem.data_ptr());
#else
  context_ =
      unique_ptr_destroy<nvinfer1::IExecutionContext>(engine_->engine->createExecutionContext());
  context_->setOptimizationProfile(profile_index_);
#endif

  // const int n_profiles = engine_->engine->getNbOptimizationProfiles();
  const int n_inputsOutputs = engine_->engine->getNbBindings() / n_profiles;

  for (int i = 0; i < n_profiles; i++) {
    for (int j = 0; j < n_inputsOutputs; j++) {
      const auto index = profile_index_ * n_inputsOutputs + j;
      if (i != profile_index_) {
        continue;
      }
      const auto name = engine_->engine->getBindingName(index);
      if (engine_->engine->bindingIsInput(index)) {
        sorted_index_inputs_[name] = index;
      } else {
        sorted_index_ouputs_[name] = index;
      }
    }
  }

  for (int i = 0; i < n_profiles; i++) {
    for (int j = 0; j < n_inputsOutputs; j++) {
      const auto index = profile_index_ * n_inputsOutputs + j;
      if (i != profile_index_) {
        continue;
      }
      const auto name = engine_->engine->getBindingName(index);
      if (engine_->engine->bindingIsInput(index)) {
        new_location_inputs_.emplace_back(
            std::distance(sorted_index_inputs_.begin(), sorted_index_inputs_.find(name)));
      } else {
        new_location_ouputs_.emplace_back(
            std::distance(sorted_index_ouputs_.begin(), sorted_index_ouputs_.find(name)));
      }
    }
  }

  std::map<std::string, std::stringstream> inputs_ss;
  std::map<std::string, std::stringstream> outputs_ss;

  // 获取输入输出尺寸信息
  for (int i = 0; i < n_profiles; i++) {
    for (int j = 0; j < n_inputsOutputs; j++) {
      const auto index = profile_index_ * n_inputsOutputs + j;
      if (i != profile_index_) {
        continue;
      }
      const auto name = engine_->engine->getBindingName(index);
      if (engine_->engine->bindingIsInput(index)) {
        nvinfer1::Dims min_dims = engine_->engine->getProfileDimensions(
            index, profile_index_, nvinfer1::OptProfileSelector::kMIN);
        nvinfer1::Dims max_dims = engine_->engine->getProfileDimensions(
            index, profile_index_, nvinfer1::OptProfileSelector::kMAX);
        mins_.emplace_back(min_dims.d, min_dims.d + min_dims.nbDims);
        maxs_.emplace_back(max_dims.d, max_dims.d + max_dims.nbDims);
        context_->setBindingDimensions(index, min_dims);

        std::stringstream ss;
        ss << "\t\t";
        for (int dim_index = 0; dim_index < min_dims.nbDims; ++dim_index) {
          ss << min_dims.d[dim_index];
          if (dim_index != min_dims.nbDims - 1) ss << "x";
        }
        ss << " -> ";
        for (int dim_index = 0; dim_index < max_dims.nbDims; ++dim_index) {
          ss << max_dims.d[dim_index];
          if (dim_index != max_dims.nbDims - 1) ss << "x";
        }
        inputs_ss[name] = std::move(ss);
        continue;
      } else {
        nvinfer1::Dims dims = context_->getBindingDimensions(index);
        std::stringstream out_s;
        out_s << "\t\t";

        for (int dim_index = 0; dim_index < dims.nbDims; ++dim_index) {
          out_s << dims.d[dim_index];
          if (dim_index != dims.nbDims - 1) out_s << "x";
        }
        outputs_ss[name] = std::move(out_s);
      }
    }
  }

  // 获取输出尺寸信息
  for (int i = 0; i < n_profiles; i++) {
    for (int j = 0; j < n_inputsOutputs; j++) {
      const auto index = profile_index_ * n_inputsOutputs + j;
      if (i != profile_index_) {
        continue;
      }
      const auto name = engine_->engine->getBindingName(index);
      if (engine_->engine->bindingIsInput(index)) {
        nvinfer1::Dims max_dims = engine_->engine->getProfileDimensions(
            index, profile_index_, nvinfer1::OptProfileSelector::kMAX);

        context_->setBindingDimensions(index, max_dims);
        continue;
      } else {
        // std::stringstream out_ss;
        nvinfer1::Dims dims = context_->getBindingDimensions(index);
        outputs_ss[name] << " -> ";
        for (int dim_index = 0; dim_index < dims.nbDims; ++dim_index) {
          outputs_ss[name] << dims.d[dim_index];
          if (dim_index != dims.nbDims - 1) outputs_ss[name] << "x";
        }
      }
    }
  }

  std::stringstream ss;
  int index = 0;
  for (const auto& item : inputs_ss) {
    ss << "\n\t\t" << index++ << ". " << item.first << "\t" << item.second.str();
  }

  index = 0;
  for (const auto& item : outputs_ss) {
    ss << "\n\t\t" << index++ << ". " << item.first << "\t" << item.second.str();
  }

  std::vector<std::vector<int>> maxs_reordered(maxs_.size());
  std::vector<std::vector<int>> mins_reordered(maxs_.size());
  for (std::size_t i = 0; i < maxs_.size(); ++i) {
    maxs_reordered[new_location_inputs_[i]] = maxs_[i];
    mins_reordered[new_location_inputs_[i]] = mins_[i];
  }
  maxs_.swap(maxs_reordered);
  mins_.swap(mins_reordered);

  if (_independent_thread_index == 0 && (n_inputsOutputs - maxs_.size() > 1 || (maxs_.size() > 1)))
    SPDLOG_INFO("Engine {}, Profile {}(Order of inputs and outputs):\n" + colored(ss.str()),
                _independent_thread_index / n_profiles, profile_index_);
  else {
    if (profile_index_ == n_profiles - 1)
      SPDLOG_INFO("Engine {}, Profile {}:\n{}", _independent_thread_index / n_profiles,
                  profile_index_,
                  colored(ss.str()));  //\t\t-------------------------------------------------
    else
      SPDLOG_INFO("Engine {}, Profile {}:\n{}", _independent_thread_index / n_profiles,
                  profile_index_, colored(ss.str()));
  }
}

decltype(at::kFloat) trt2torch_type(decltype(nvinfer1::DataType::kFLOAT) dtype) {
  auto target_dtype = at::kFloat;
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      target_dtype = at::kFloat;
      break;
    case nvinfer1::DataType::kINT32:
      target_dtype = at::kInt;
      break;
    case nvinfer1::DataType::kINT8:
      target_dtype = at::kChar;
      break;
// case nvinfer1::DataType::kUINT8:
//   target_dtype = at::kByte;
//   break;
#if NV_TENSORRT_MAJOR >= 9
    case nvinfer1::DataType::kINT64:
      target_dtype = at::kLong;
      break;
#endif

    case nvinfer1::DataType::kBOOL:
      target_dtype = at::kBool;
      break;
    case nvinfer1::DataType::kHALF:
      target_dtype = at::kHalf;
      break;
    default:
      SPDLOG_ERROR("out: only support type of kFLOAT, kINT32, kINT64, kINT8, kBOOL, kHALF");
      throw std::runtime_error("unsupportted datatype");
  }
  return target_dtype;
}

at::Tensor guard_contiguous_type_and_device(at::Tensor input_data, at::ScalarType target_format) {
  // assert(input_data.is_cuda() && input_data.scalar_type() == at::kFloat);
  if (target_format == input_data.dtype() && !is_cpu_tensor(input_data) &&
      input_data.is_contiguous()) {
    return input_data;
  }

  input_data = input_data.to(at::kCUDA, target_format,
                             /* non_blocking =*/false, false, at::MemoryFormat::Contiguous);
  if (!input_data.is_contiguous()) {
    input_data = input_data.contiguous();
  }
  return input_data;
}
void TensorrtTensor::forward(const std::vector<dict>& raw_inputs) {
  assert(raw_inputs.size() <= max() && raw_inputs.size() >= min());
  if (raw_inputs.empty()) return;

  auto node_name = dict_get<std::string>(raw_inputs[0], "node_name", true);

  {
    inputs_ = preprocessor_->forward(raw_inputs);
    if (inputs_.empty()) {
      return;
    }
  }
#ifndef NDEBUG
  std::string shape = std::to_string(inputs_[0].size(0));
  for (std::size_t i = 1; i < inputs_[0].sizes().size(); ++i) {
    shape += " " + std::to_string(inputs_[0].size(i));
  }

  SPDLOG_DEBUG("{}: index({}) size({}) max-{} input.shape({})", node_name.c_str(),
               independent_thread_index_, raw_inputs.size(), max(), shape);
#endif

  {
    for (auto& true_input : inputs_) {
    }
  }

  binding_.clear();
  outputs_.clear();

  const unsigned n_profiles = engine_->engine->getNbOptimizationProfiles();
  const unsigned n_inputsOutputs = engine_->engine->getNbBindings() / n_profiles;

  int index_input = 0;

  for (unsigned i = 0; i < n_profiles; i++) {
    for (unsigned j = 0; j < n_inputsOutputs; j++) {
      if (i != profile_index_) {
        binding_.push_back(nullptr);
        continue;
      }
      const auto index = profile_index_ * n_inputsOutputs + j;

      nvinfer1::Dims infer_dims = context_->getBindingDimensions(index);
      auto dtype = engine_->engine->getBindingDataType(index);
      auto target_type = trt2torch_type(dtype);

      if (engine_->engine->bindingIsInput(index)) {
        auto& input_data = inputs_[new_location_inputs_[index_input]];

        input_data = guard_contiguous_type_and_device(input_data, target_type);

        binding_.push_back(input_data.data_ptr());
        if (infer_dims.nbDims != input_data.sizes().size()) {
          std::stringstream ss;
          ss << "shape not match: model input and preprocessed tensor(s): (infer_dims)"
             << infer_dims.nbDims << " != (input_data)" << input_data.sizes().size();
          SPDLOG_ERROR(ss.str());
          throw std::runtime_error(ss.str());
        }

        for (decltype(infer_dims.nbDims) t = 0; t < infer_dims.nbDims; ++t) {
          if (infer_dims.d[t] != input_data.size(t)) {
            const auto& profile_mins_shape = mins_[new_location_inputs_[index_input]];
            const auto& profile_maxs_shape = maxs_[new_location_inputs_[index_input]];

            if (input_data.size(t) < profile_mins_shape[t] ||
                input_data.size(t) > profile_maxs_shape[t]) {
              std::stringstream ss;
              ss << "shape out of range: input_data.size(" << t << "): " << input_data.size(t)
                 << ", input range is [" << profile_mins_shape[t] << ", " << profile_maxs_shape[t]
                 << "]";
              SPDLOG_ERROR(ss.str());
              throw std::runtime_error(ss.str());
            }
            change_shape_[index_input] = true;
            infer_dims.d[t] = input_data.size(t);
          }
        }

        if (change_shape_[index_input]) {
          context_->setBindingDimensions(index, infer_dims);
          change_shape_[index_input] = false;
        }

        index_input++;
        continue;
      }

      // infer_dims = context_->getBindingDimensions(index);

      if (infer_dims.nbDims == -1) {
        throw std::range_error("tensorrt: getBindingDimensions for output failed");
      }

      outputs_.emplace_back(
          at::empty(std::vector<int64_t>(infer_dims.d, infer_dims.d + infer_dims.nbDims),
                    get_tensor_option(target_type), at::MemoryFormat::Contiguous));
      binding_.push_back(outputs_.back().data_ptr());
    }
  }
#ifdef USE_OUT_MEM
  auto mem = torch_allocate(engine_->engine->getDeviceMemorySize());
  context_->setDeviceMemory(mem.data_ptr());
#endif

  { context_->enqueueV2(&binding_[0], c10::cuda::getCurrentCUDAStream(), nullptr); }

  auto outputs_sorted = std::vector<at::Tensor>(outputs_.size());
  for (std::size_t i = 0; i < outputs_.size(); ++i) {
    outputs_sorted[new_location_ouputs_[i]] = outputs_[i];
  }
  outputs_.swap(outputs_sorted);

  postprocessor_->forward(outputs_, raw_inputs, inputs_);

  at::cuda::getCurrentCUDAStream().synchronize();  // for Reclaim GPU Memory of mem outputs_
  outputs_.clear();
  inputs_.clear();
}

IPIPE_REGISTER(Backend, TensorrtTensor, "TensorrtTensor");

}  // namespace ipipe
#endif