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

#ifdef WITH_TENSORRT

#include "tensorrt_utils.hpp"
// #if NV_TENSORRT_MAJOR >= 9
#if (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 5) || (NV_TENSORRT_MAJOR >= 9)
#include <torch/torch.h>
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
#include "TensorrtTensor.hpp"
#include "Onnx2TensorrtConverter.hpp"
#include "SingleConcatPreprocess.hpp"
#include "MultipleConcatPreprocess.hpp"
// https://github.com/NVIDIA/Torch-TensorRT/blob/3a98a8b198a071e622c43283caea7416fe8a8a1a/core/runtime/register_trt_op.cpp

// #define USE_OUT_MEM
namespace ipipe {
// record stream;
// https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486/5

bool TensorrtTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                          dict dict_config) {
  SPDLOG_WARN(
      "TensorRT <= 8.4 is deprecated, and torchpipe behavior follows the old version "
      "mode, which may be slightly different from the behavior in the higher version mode. Please "
      "use TensorRT >= 8.5. Got {}.{}",
      NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
#if NV_TENSORRT_MAJOR < 8
  SPDLOG_ERROR("tensorrt version should >= 8 but got {}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
  return false;
#endif
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
                                                {"force_range", ""},
                                                {"batch_process", ""},
                                                {"input_reorder", ""},
                                                {"output_reorder", ""}},
                                               {}, {}, {}));

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

  // unique_ptr_destroy<nvinfer1::IHostMemory> p_engine_plan{engine_->engine->serialize()};

  // std::string engine_plan = std::string((char*)p_engine_plan->data(), p_engine_plan->size());

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

  postprocessor_ = std::unique_ptr<PostProcessor<torch::Tensor>>(
      IPIPE_CREATE(PostProcessor<torch::Tensor>, batch_post));
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
      preprocessor_ = std::unique_ptr<PreProcessor<torch::Tensor>>(new SingleConcatPreprocess());
    } else {
      preprocessor_ = std::unique_ptr<PreProcessor<torch::Tensor>>(new MultipleConcatPreprocess());
    }
  } else {
    preprocessor_ = std::unique_ptr<PreProcessor<torch::Tensor>>(
        IPIPE_CREATE(PreProcessor<torch::Tensor>, preprocessor));
  }

  (*dict_config)["max"] = maxs_;
  (*dict_config)["min"] = mins_;

  if (!preprocessor_ || !preprocessor_->init(config_param, dict_config)) {
    SPDLOG_ERROR("preprocess_engine created({}) or inited failed .", bool(preprocessor_));
    return false;
  }

  // save engne if needed.
  if (!params_->at("input_reorder").empty()) {
    input_reorder_ = str2int(params_->at("input_reorder"), ',');
  }
  if (!params_->at("output_reorder").empty()) {
    output_reorder_ = str2int(params_->at("output_reorder"), ',');
  }

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
  if (!params_->at("batch_process").empty()) {
    batch_process_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("batch_process")));
    IPIPE_ASSERT(batch_process_->init(config_param, dict_config) && batch_process_->min() == 1);
  }

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
  context_->setOptimizationProfileAsync(profile_index_, c10::cuda::getCurrentCUDAStream());
#endif

  // const int n_profiles = engine_->engine->getNbOptimizationProfiles();
  // const int n_inputsOutputs = engine_->engine->getNbBindings() / n_profiles;
  // tensorrt 10:
  const int n_inputsOutputs = engine_->engine->getNbIOTensors();

  int n_inputs = 0;
  int n_outputs = 0;
  for (int j = 0; j < n_inputsOutputs; j++) {
    const auto name = engine_->engine->getIOTensorName(j);
    const auto tensorType = engine_->engine->getTensorIOMode(name);
    if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
      n_inputs += 1;
    } else {
      n_outputs += 1;
      IPIPE_ASSERT(j >= n_inputs);
    }
  }

  auto reorder_by_alpha = [](std::vector<int>& reorder,
                             std::shared_ptr<CudaEngineWithRuntime> engine, int offset) {
    std::map<std::string, int> name2index;
    for (size_t index = 0; index < reorder.size(); ++index) {
      name2index[engine->engine->getIOTensorName(index + offset)] = index;
    }
    auto name2index_it = name2index.begin();
    for (size_t i = 0; name2index_it != name2index.end(); ++i, ++name2index_it) {
      reorder[i] = name2index_it->second;
    }
  };
  IPIPE_ASSERT(n_inputs + n_outputs == n_inputsOutputs);
  if (!input_reorder_.empty()) {
    IPIPE_ASSERT(input_reorder_.size() == n_inputs);
  } else {
    // [DEAFULT] set input_reorder_ to n_inputs-1, ... , 0
    input_reorder_.resize(n_inputs);
// std::iota(input_reorder_.rbegin(), input_reorder_.rend(), 0);
// std::iota(input_reorder_.begin(), input_reorder_.end(), 0);
#if NV_TENSORRT_MAJOR >= 8
    std::iota(input_reorder_.begin(), input_reorder_.end(), 0);
#else
    reorder_by_alpha(input_reorder_, engine_, 0);
#endif
  }
  if (!output_reorder_.empty()) {
    IPIPE_ASSERT(output_reorder_.size() == n_outputs);
  } else {
    output_reorder_.resize(n_outputs);
//
#if NV_TENSORRT_MAJOR >= 8
    std::iota(output_reorder_.begin(), output_reorder_.end(), 0);
#else
    reorder_by_alpha(output_reorder_, engine_, input_reorder_.size());
#endif
  }

  std::vector<std::pair<std::string, std::string>> inputs_ss(input_reorder_.size());
  std::vector<std::pair<std::string, std::string>> outputs_ss(output_reorder_.size());

  // 获取输入输出尺寸信息
  for (int j = 0; j < input_reorder_.size(); j++) {
    // tensorrt is inverted order
    const auto name = engine_->engine->getIOTensorName(input_reorder_[j]);
    const auto tensorType = engine_->engine->getTensorIOMode(name);

    IPIPE_ASSERT(tensorType == nvinfer1::TensorIOMode::kINPUT);

#if 1
    nvinfer1::Dims min_dims =
        engine_->engine->getProfileShape(name, profile_index_, nvinfer1::OptProfileSelector::kMIN);
    nvinfer1::Dims max_dims =
        engine_->engine->getProfileShape(name, profile_index_, nvinfer1::OptProfileSelector::kMAX);
#else
    std::string post_name;
    if (profile_index_) {
      post_name + " [profile " + std::to_string(profile_index_) + "]";
    }
    nvinfer1::Dims min_dims = engine_->engine->getProfileShape(name + post_name, profile_index_,
                                                               nvinfer1::OptProfileSelector::kMIN);
    nvinfer1::Dims max_dims = engine_->engine->getProfileShape(name + post_name, profile_index_,
                                                               nvinfer1::OptProfileSelector::kMAX);
#endif

    mins_.emplace_back(min_dims.d, min_dims.d + min_dims.nbDims);
    maxs_.emplace_back(max_dims.d, max_dims.d + max_dims.nbDims);
    IPIPE_ASSERT(context_->setInputShape(name, min_dims));

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
    inputs_ss[input_reorder_[j]] = {name, ss.str()};
  }

  for (int j = 0; j < output_reorder_.size(); j++) {
    // tensorrt is inverted order
    const auto name = engine_->engine->getIOTensorName(input_reorder_.size() + output_reorder_[j]);
    const auto tensorType = engine_->engine->getTensorIOMode(name);

    nvinfer1::Dims dims = context_->getTensorShape(name);
    std::stringstream ss;
    ss << "\t\t";

    for (int dim_index = 0; dim_index < dims.nbDims; ++dim_index) {
      ss << dims.d[dim_index];
      if (dim_index != dims.nbDims - 1) ss << "x";
    }
    // outputs_ss[name] = out_s;
    outputs_ss[output_reorder_[j]] = {name, ss.str()};
  }

  for (int j = 0; j < input_reorder_.size(); j++) {
    // tensorrt is inverted order
    const auto name = engine_->engine->getIOTensorName(input_reorder_[j]);
    // #ifdef NV_TENSORRT_MAJOR> 8
    nvinfer1::Dims max_dims =
        engine_->engine->getProfileShape(name, profile_index_, nvinfer1::OptProfileSelector::kMAX);
    // #else
    // todo
    // #endif
    IPIPE_ASSERT(context_->setInputShape(name, max_dims));
  }

  // 获取max输出尺寸信息
  for (int j = 0; j < output_reorder_.size(); j++) {
    const auto name = engine_->engine->getIOTensorName(input_reorder_.size() + output_reorder_[j]);
    // const auto tensorType = engine_->engine->getTensorIOMode(name);

    // if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
    //   // nvinfer1::Dims max_dims = engine_->engine->getProfileShape(
    //   //     name, profile_index_, nvinfer1::OptProfileSelector::kMAX);

    //   // context_->setInputShape(name, max_dims);
    //   // continue;
    // } else

    // std::stringstream out_ss;
    nvinfer1::Dims dims = context_->getTensorShape(name);
    outputs_ss[output_reorder_[j]].second += " -> ";
    for (int dim_index = 0; dim_index < dims.nbDims; ++dim_index) {
      outputs_ss[output_reorder_[j]].second += std::to_string(dims.d[dim_index]);
      if (dim_index != dims.nbDims - 1) outputs_ss[output_reorder_[j]].second += "x";
    }
  }

  std::string current_order = "(current: ";
  std::stringstream ss;
  for (std::size_t i = 0; i < inputs_ss.size(); ++i) {
    ss << "\n\t\t" << input_reorder_[i] << ". " << inputs_ss[i].first << "\t"
       << inputs_ss[i].second;
    current_order += std::to_string(input_reorder_[i]) + ",";
  }
  current_order.pop_back();
  current_order += " reset by `input_reorder`)";

  for (std::size_t i = 0; i < output_reorder_.size(); ++i) {
    ss << "\n\t\t" << output_reorder_[i] << ". " << outputs_ss[output_reorder_[i]].first << "\t"
       << outputs_ss[output_reorder_[i]].second;
  }

  // for (std::size_t i = 0; i < outputs_ss.size(); ++i) {
  //   ss << "\n\t\t" << output_reorder_[i] << ". " << outputs_ss[i].first << "\t"
  //      << outputs_ss[i].second;
  // }

  if (_independent_thread_index == 0 && (n_inputsOutputs - maxs_.size() > 1 || (maxs_.size() > 1)))
    SPDLOG_INFO("Engine {}, Profile {}: Order of inputs{} and outputs:\n" + colored(ss.str()),
                _independent_thread_index / n_profiles, profile_index_, current_order.c_str());
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

decltype(torch::kFloat) trt2torch_type(decltype(nvinfer1::DataType::kFLOAT) dtype) {
  auto target_dtype = torch::kFloat;
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      target_dtype = torch::kFloat;
      break;
    case nvinfer1::DataType::kINT32:
      target_dtype = torch::kInt;
      break;
    case nvinfer1::DataType::kINT8:
      target_dtype = torch::kChar;
      break;
// case nvinfer1::DataType::kUINT8:
//   target_dtype = torch::kByte;
//   break;
#if NV_TENSORRT_MAJOR >= 9
    case nvinfer1::DataType::kINT64:
      target_dtype = torch::kLong;
      break;
#endif

    case nvinfer1::DataType::kBOOL:
      target_dtype = torch::kBool;
      break;
    case nvinfer1::DataType::kHALF:
      target_dtype = torch::kHalf;
      break;
    default:
      SPDLOG_ERROR("out: only support type of kFLOAT, kINT32, kINT64, kINT8, kBOOL, kHALF");
      throw std::runtime_error("unsupportted datatype");
  }
  return target_dtype;
}

torch::Tensor guard_contiguous_type_and_device(torch::Tensor input_data,
                                               torch::ScalarType target_format) {
  // assert(input_data.is_cuda() && input_data.scalar_type() == torch::kFloat);
  if (target_format == input_data.dtype() && !is_cpu_tensor(input_data) &&
      input_data.is_contiguous()) {
    return input_data;
  }

  input_data = input_data.to(torch::kCUDA, target_format,
                             /* non_blocking =*/false, false, torch::MemoryFormat::Contiguous);
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
    IPIPE_ASSERT(!inputs_.empty());
  }
#ifndef NDEBUG
  std::string shape;
  for (const auto& input : inputs_) {
    shape += std::to_string(input.size(0));

    for (std::size_t i = 1; i < input.sizes().size(); ++i) {
      shape += "x" + std::to_string(input.size(i));
    }
    shape += ",";
  }
  shape.pop_back();

  SPDLOG_DEBUG("{}: instance({}) dicts({}) this->max()={}({})", node_name.c_str(),
               independent_thread_index_, raw_inputs.size(), max(), shape);
#endif

  binding_.clear();
  outputs_.clear();

  const unsigned n_profiles = engine_->engine->getNbOptimizationProfiles();
  // const unsigned n_inputsOutputs = engine_->engine->getNbBindings() / n_profiles;
  const unsigned n_inputsOutputs = engine_->engine->getNbIOTensors();

  // int index_input = 0;

  for (unsigned j = 0; j < input_reorder_.size(); j++) {
    const auto name = engine_->engine->getIOTensorName(input_reorder_[j]);
    const auto tensorType = engine_->engine->getTensorIOMode(name);

    nvinfer1::Dims infer_dims = context_->getTensorShape(name);
    auto dtype = engine_->engine->getTensorDataType(name);
    auto target_type = trt2torch_type(dtype);

    IPIPE_ASSERT(tensorType == nvinfer1::TensorIOMode::kINPUT);

    int index_input = j;  // todo
    auto& input_data = inputs_[j];

    input_data = guard_contiguous_type_and_device(input_data, target_type);

    if (infer_dims.nbDims != input_data.sizes().size()) {
      std::stringstream ss;
      ss << "shape not match: model input and preprocessed tensor(s): (infer_dims)"
         << infer_dims.nbDims << " != (input_data)" << input_data.sizes().size();
      SPDLOG_ERROR(ss.str());
      throw std::runtime_error(ss.str());
    }

    const auto& profile_mins_shape = mins_[index_input];
    const auto& profile_maxs_shape = maxs_[index_input];
    for (decltype(infer_dims.nbDims) t = 0; t < infer_dims.nbDims; ++t) {
      if (infer_dims.d[t] != input_data.size(t)) {
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
      context_->setInputShape(name, infer_dims);
      change_shape_[index_input] = false;
    }
    bool status = context_->setTensorAddress(name, input_data.data_ptr());
    IPIPE_ASSERT(status);
  }

  auto iter_outputs = raw_inputs[0]->find("outputs");
  std::vector<torch::Tensor> predefined_outputs;
  if (iter_outputs != raw_inputs[0]->end()) {
    IPIPE_ASSERT(raw_inputs.size() == 1);
    predefined_outputs = any_cast<std::vector<torch::Tensor>>(iter_outputs->second);
    IPIPE_ASSERT(!predefined_outputs.empty());
  }

  for (unsigned j = 0; j < output_reorder_.size(); j++) {
    const auto name = engine_->engine->getIOTensorName(input_reorder_.size() + output_reorder_[j]);
    const auto tensorType = engine_->engine->getTensorIOMode(name);
    IPIPE_ASSERT(tensorType == nvinfer1::TensorIOMode::kOUTPUT);

    nvinfer1::Dims infer_dims = context_->getTensorShape(name);
    auto dtype = engine_->engine->getTensorDataType(name);
    auto target_type = trt2torch_type(dtype);

    if (infer_dims.nbDims == -1) {
      throw std::range_error("tensorrt: getBindingDimensions for output failed");
    }

    if (predefined_outputs.size() > j) {
      IPIPE_ASSERT(predefined_outputs[j].is_contiguous());
      int64_t total_bytes = predefined_outputs[j].numel() * predefined_outputs[j].element_size();
      int64_t need_bytes = std::accumulate(infer_dims.d, infer_dims.d + infer_dims.nbDims, 1,
                                           std::multiplies<int64_t>()) *
                           torch::elementSize(target_type);
      IPIPE_ASSERT(need_bytes == total_bytes);

      outputs_.emplace_back(predefined_outputs[j]);
    } else {
      outputs_.emplace_back(
          torch::empty(std::vector<int64_t>(infer_dims.d, infer_dims.d + infer_dims.nbDims),
                       get_tensor_option(target_type), torch::MemoryFormat::Contiguous));
    }

    bool status = context_->setTensorAddress(name, outputs_.back().data_ptr());
    IPIPE_ASSERT(status);
  }

#ifdef USE_OUT_MEM
  auto mem = torch_allocate(engine_->engine->getDeviceMemorySize());
  context_->setDeviceMemory(mem.data_ptr());
#endif

  IPIPE_ASSERT(context_->enqueueV3(c10::cuda::getCurrentCUDAStream()));

  if (batch_process_) {
    dict batched = std::make_shared<std::unordered_map<std::string, any>>();
    (*batched)[TASK_DATA_KEY] = outputs_;
    batch_process_->forward({batched});
    TRACE_EXCEPTION(outputs_ = any_cast<std::vector<torch::Tensor>>(batched->at(TASK_RESULT_KEY)));
  }

  postprocessor_->forward(outputs_, raw_inputs, inputs_);

  c10::cuda::getCurrentCUDAStream().synchronize();
  outputs_.clear();  // for Reclaim GPU Memory of mem outputs_
  inputs_.clear();
}

IPIPE_REGISTER(Backend, TensorrtTensor, "TensorrtTensor");

}  // namespace ipipe
#endif
#endif