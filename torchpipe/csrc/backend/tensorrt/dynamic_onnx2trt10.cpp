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

#include <NvInferRuntime.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"

#if NV_TENSORRT_MAJOR >= 10
#include "time_utils.hpp"
#include "tensorrt_utils.hpp"
#include "ipipe_common.hpp"
#include "dynamic_onnx2trt.hpp"
#include "base_logging.hpp"
#include "decrypt.hpp"
#include "dict.hpp"
#include "params.hpp"
#include "Calibrator.hpp"
#include <cuda.h>
#include "c10/cuda/CUDAStream.h"
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include "Backend.hpp"

#ifdef USE_TORCH_ALLOCATOR
#include "torch_allocator.hpp"
#endif

namespace ipipe {
nvinfer1::Dims vtodim(std::vector<int> input) {
  nvinfer1::Dims out;
  out.nbDims = input.size();
  for (std::size_t i = 0; i < input.size(); ++i) out.d[i] = input[i];

  return out;
}

nvinfer1::Dims shape_tensor_to_dim(std::vector<int> input) {
  IPIPE_ASSERT(input.size() >= 1);
  nvinfer1::Dims out;
  out.nbDims = input.size();
  for (std::size_t i = 0; i < input.size(); ++i) out.d[i] = input[i];
  return out;
}

nvinfer1::ILogger::Severity trt_get_log_level(std::string level) {
  if (level == "info")
    return nvinfer1::ILogger::Severity::kINFO;
  else if (level == "error")
    return nvinfer1::ILogger::Severity::kERROR;
  else if (level == "verbose")
    return nvinfer1::ILogger::Severity::kVERBOSE;
  else
    return nvinfer1::ILogger::Severity::kWARNING;
  throw std::invalid_argument("log_level must be one of info, error, verbose, warning");
}

nvinfer1::Dims vtodim(std::vector<int> input, const nvinfer1::Dims& net_input) {
  nvinfer1::Dims out = net_input;
  // out.nbDims = input.size();
  if (input.size() < net_input.nbDims) {
    input.resize(net_input.nbDims, -1);
  }
  for (std::size_t i = 0; i < net_input.nbDims; ++i) {
    if (input[i] == -1) {
      if (-1 == net_input.d[i]) {
        const std::string error_msg =
            "shape from network and shape from configuration not match: net_input.d[i]= " +
            std::to_string(net_input.d[i]) + " input[i]= " + std::to_string(input[i]);
        SPDLOG_ERROR(error_msg);
        throw std::invalid_argument(error_msg);
      } else {
        input[i] = net_input.d[i];
      }
    } else if (net_input.d[i] != -1 && net_input.d[i] != input[i]) {
      const std::string error_msg =
          "shape from network and shape from configuration not match: net_input[" +
          std::to_string(i) + "]= " + std::to_string(net_input.d[i]) +
          " input= " + std::to_string(input[i]);
      SPDLOG_ERROR(error_msg);
      throw std::invalid_argument(error_msg);
    }
    out.d[i] = input[i];
  }

  return out;
}

bool Is_File_Exist(const std::string& file_path) {
  std::ifstream file(file_path.c_str());
  return file.good();
}

class NvLogger : public nvinfer1::ILogger {
 public:
#if NV_TENSORRT_MAJOR < 8
  void log(Severity severity, const char* msg) override {
    // remove this 'if' if you need more logged info
    if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
      SPDLOG_ERROR(msg);
    } else if (severity == Severity::kINFO)
      SPDLOG_INFO(msg);
    else if (severity == Severity::kWARNING)
      SPDLOG_WARN(msg);
    else if (severity == Severity::kVERBOSE)
      SPDLOG_TRACE(msg);
  }
#else
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
    if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
      // SPDLOG_ERROR(msg);
      spdlog::error(msg);
    } else if (severity == Severity::kINFO)
      // SPDLOG_DEBUG(msg);
      spdlog::debug(msg);
    else if (severity == Severity::kWARNING)
      // SPDLOG_WARN(msg);
      spdlog::warn(msg);
    else if (severity == Severity::kVERBOSE)
      // SPDLOG_TRACE(msg);
      spdlog::trace(msg);
  }
#endif
} gLogger_inplace;

std::string getBasename(std::string const& path) {
#ifdef _WIN32
  constexpr char SEPARATOR = '\\';
#else
  constexpr char SEPARATOR = '/';
#endif
  int baseId = path.rfind(SEPARATOR) + 1;
  return path.substr(baseId, path.rfind('.') - baseId);
}

bool initPlugins() {
  static bool didInitPlugins = initLibNvInferPlugins(&gLogger_inplace, "");
  assert(didInitPlugins);
  return didInitPlugins;
}

std::vector<std::vector<int>> infer_onnx_shape(std::string onnx_path) {
  constexpr auto explicitBatch =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  unique_ptr_destroy<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger_inplace)};

  unique_ptr_destroy<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
  unique_ptr_destroy<nvonnxparser::IParser> parser{
      nvonnxparser::createParser(*network, gLogger_inplace)};
  unique_ptr_destroy<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};

  // 获取模型输入大小
  auto parsed = parser->parseFromFile(onnx_path.c_str(),
                                      static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
  if (!parsed) {
    throw std::runtime_error("Failed to parse onnx file");
  }
  std::vector<std::vector<int>> ret;
  for (auto input_index = 0; input_index < network->getNbInputs(); input_index++) {
    auto input = network->getInput(input_index);
    auto dims = input->getDimensions();
    std::vector<int> shape;
    for (int i = 0; i < dims.nbDims; i++) {
      shape.push_back(dims.d[i]);
    }
    ret.emplace_back(shape);
  }
  return ret;
}

std::shared_ptr<CudaEngineWithRuntime> loadEngineFromBuffer(const std::string& engine_plan) {
  std::shared_ptr<CudaEngineWithRuntime> en_with_rt =
      std::make_shared<CudaEngineWithRuntime>(nvinfer1::createInferRuntime(gLogger_inplace));
#ifdef USE_TORCH_ALLOCATOR
  const char* value = std::getenv("PYTORCH_NO_CUDA_MEMORY_CACHING");
  bool using_default_stream =
      c10::cuda::getCurrentCUDAStream() == c10::cuda::getDefaultCUDAStream();
  if (value == nullptr && !using_default_stream) {
    SPDLOG_INFO("use torch allocator");
    en_with_rt->allocator = new TorchAllocator();
    en_with_rt->runtime->setGpuAllocator(en_with_rt->allocator);
  }
#endif

  IPIPE_ASSERT(en_with_rt && en_with_rt->deserializeCudaEngine(engine_plan));
  return en_with_rt;
}

std::shared_ptr<CudaEngineWithRuntime> loadCudaBackend(std::string const& trtModelPath,
                                                       const std::string& model_type,
                                                       std::string& engine_plan) {
  std::shared_ptr<CudaEngineWithRuntime> en_with_rt =
      std::make_shared<CudaEngineWithRuntime>(nvinfer1::createInferRuntime(gLogger_inplace));

#ifdef USE_TORCH_ALLOCATOR
  const char* value = std::getenv("PYTORCH_NO_CUDA_MEMORY_CACHING");
  bool using_default_stream =
      c10::cuda::getCurrentCUDAStream() == c10::cuda::getDefaultCUDAStream();
  if (value == nullptr && !using_default_stream) {
    SPDLOG_INFO("use torch allocator");
    en_with_rt->allocator = new TorchAllocator();
    en_with_rt->runtime->setGpuAllocator(en_with_rt->allocator);
  }
#endif

  std::vector<char> trtModelStream;
  if (model_type == ".trt.encrypt") {
    bool succ = aes_256_decrypt_from_path(trtModelPath, trtModelStream);
    if (!succ) {
      return nullptr;
    }
  } else if (model_type == ".trt") {
    std::ifstream file(trtModelPath, std::ios::binary);
    if (file.good()) {
      size_t size{0};
      file.seekg(0, file.end);
      size = file.tellg();
      file.seekg(0, file.beg);
      trtModelStream.resize(size);
      file.read(trtModelStream.data(), size);
      file.close();
    } else {
      SPDLOG_ERROR(trtModelPath + " not exists.\n");
      return nullptr;
    }

  } else {
    SPDLOG_ERROR(trtModelPath + " : not support this model!!\n");
    return nullptr;
  }

  IPIPE_ASSERT(en_with_rt &&
               en_with_rt->deserializeCudaEngine(trtModelStream.data(), trtModelStream.size()));
  engine_plan = std::string(trtModelStream.begin(), trtModelStream.end());
  assert(!engine_plan.empty());
  return en_with_rt;
}

nvinfer1::ITimingCache* prepareTimeCache(const std::string& cache_file,
                                         nvinfer1::IBuilderConfig* config) {
  std::vector<char> cache;
  if (!Is_File_Exist(cache_file)) {
    nvinfer1::ITimingCache* timingCache =
        config->createTimingCache(static_cast<const void*>(cache.data()), cache.size());
    IPIPE_ASSERT(config->setTimingCache(*timingCache, false));
    return timingCache;
  }
  std::ifstream iFile(cache_file, std::ios::in | std::ios::binary);

  if (!iFile) {
    throw std::runtime_error("Could not read timing cache from: " + cache_file);
  } else {
    iFile.seekg(0, std::ifstream::end);
    size_t fsize = iFile.tellg();
    iFile.seekg(0, std::ifstream::beg);
    cache.resize(fsize);
    iFile.read(cache.data(), fsize);
    iFile.close();

    SPDLOG_INFO("Load {}K of timing cache from {}", cache.size() / 1000.0, cache_file);
  }

  nvinfer1::ITimingCache* timingCache =
      config->createTimingCache(static_cast<const void*>(cache.data()), cache.size());
  IPIPE_ASSERT(config->setTimingCache(*timingCache, false));
  return timingCache;
}

void writeTimeCache(const std::string& cache_file, nvinfer1::IBuilderConfig* config,
                    nvinfer1::ITimingCache* timingCache) {
  // fileTimingCache->combine(*timingCache, false);
  auto blob = std::unique_ptr<nvinfer1::IHostMemory>(config->getTimingCache()->serialize());

  if (!blob || !blob->size()) return;

  if (Is_File_Exist(cache_file)) {
    // 如果cache_file的大小和blob的大小一样，就不用写了
    std::ifstream iFile(cache_file, std::ios::in | std::ios::binary);
    if (iFile) {
      iFile.seekg(0, std::ifstream::end);
      size_t fsize = iFile.tellg();
      iFile.seekg(0, std::ifstream::beg);
      if (fsize == blob->size()) {
        SPDLOG_INFO("size of {} is same as blob's size, skip writing timing cache.", cache_file);
        return;
      }
    }
  }

  if (!blob) {
    throw std::runtime_error("Failed to serialize ITimingCache!");
  }
  std::ofstream oFile(cache_file, std::ios::out | std::ios::binary);
  IPIPE_ASSERT(oFile);

  oFile.write((char*)blob->data(), blob->size());
  oFile.close();
  SPDLOG_INFO("Saved {} K of timing cache to {}", blob->size() / 1000., cache_file);
}

std::shared_ptr<CudaEngineWithRuntime> onnx2trt(
    std::string const& onnxModelPath, std::string model_type,
    std::vector<std::vector<std::vector<int>>>&
        mins,  // multiple profiles - multiple inputs - multiDims
    std::vector<std::vector<std::vector<int>>>& maxs, std::string& engine_plan,
    const OnnxParams& precision, const std::unordered_map<std::string, std::string>& config_param,
    std::vector<float> _mean, std::vector<float> _std) {
  if (!endswith(model_type, ".buffer") && !Is_File_Exist(onnxModelPath)) {
    SPDLOG_ERROR(onnxModelPath + " not exists.\n\n");
    return nullptr;
  }
  auto input_reorder = precision.input_reorder;

  const static std::unordered_set<std::string> int8_enable{"int8", "best"};
  const static std::unordered_set<std::string> fp16_enable{"fp16", "int8", "best"};

  constexpr auto explicitBatch =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  unique_ptr_destroy<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger_inplace)};

#ifdef USE_TORCH_ALLOCATOR
  const char* value = std::getenv("PYTORCH_NO_CUDA_MEMORY_CACHING");
  std::unique_ptr<TorchAllocator> allocator = std::make_unique<TorchAllocator>();
  bool using_default_stream =
      c10::cuda::getCurrentCUDAStream() == c10::cuda::getDefaultCUDAStream();
  if (value == nullptr && allocator && !using_default_stream) {
    SPDLOG_INFO("use torch allocator");
    builder->setGpuAllocator(allocator.get());
  }

#endif

  unique_ptr_destroy<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
  unique_ptr_destroy<nvonnxparser::IParser> parser{
      nvonnxparser::createParser(*network, gLogger_inplace)};
  unique_ptr_destroy<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};

#if ((NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >= 4) || (NV_TENSORRT_MAJOR >= 9))
  auto hardware_concurrency = std::thread::hardware_concurrency();
  if (hardware_concurrency == 0) hardware_concurrency = 4;
  if (hardware_concurrency >= 8) hardware_concurrency = 4;
  builder->setMaxThreads(hardware_concurrency);
  SPDLOG_INFO("nvinfer1::IBuilder: setMaxThreads {}.", hardware_concurrency);
#endif

#if NV_TENSORRT_MAJOR >= 8
  std::unique_ptr<nvinfer1::ITimingCache> time_cache;
  if (!precision.timecache.empty()) {
    time_cache.reset(prepareTimeCache(precision.timecache, config.get()));
  }
#else
  IPIPE_ASSERT(precision.timecache.empty());
#endif

#if CUDA_VERSION <= 10020

  // https://github.com/NVIDIA/TensorRT/issues/866
  auto tacticSources = static_cast<uint32_t>(config->getTacticSources());
  // kCUBLAS kCUBLAS_LT kCUDNN kEDGE_MASK_CONVOLUTIONS
  tacticSources &= ~(1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS_LT));
  config->setTacticSources(static_cast<nvinfer1::TacticSources>(tacticSources));
#endif

  bool b_parsed = false;

  if (endswith(model_type, ".onnx")) {
    // SPDLOG_INFO("start parsing {}", onnxModelPath);
    b_parsed = parser->parseFromFile(onnxModelPath.c_str(),
                                     static_cast<int>(trt_get_log_level(precision.log_level)));
  } else if (endswith(model_type, ".onnx.encrypt")) {
    std::vector<char> onnx_data_result;
    bool succ = aes_256_decrypt_from_path(onnxModelPath, onnx_data_result);
    if (succ) {
      b_parsed = parser->parse(onnx_data_result.data(), onnx_data_result.size());
    }
  } else if (endswith(model_type, ".onnx.buffer")) {
    b_parsed = parser->parse(onnxModelPath.data(), onnxModelPath.size());
  } else {
    if (model_type.size() > 100) {
      model_type = model_type.substr(model_type.size() - 50);
    }
    SPDLOG_ERROR("wrong model_type: {}", model_type);
  }

  if (b_parsed) {
    bool use_only_fp32 = true;
    // constexpr size_t MAX_WORKSPACE_SIZE = 1ULL << 30;  // 1 GB
#if (NV_TEONSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4) || (NV_TENSORRT_MAJOR >= 9)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, precision.max_workspace_size);
#else
    config->setMaxWorkspaceSize(precision.max_workspace_size);
#endif
    if (precision.max_workspace_size != 1024)
      SPDLOG_INFO("max workspace size setted to {}M",
                  precision.max_workspace_size / 1024.0 / 1024.0);
    if ((fp16_enable.count(precision.precision)) && builder->platformHasFastFp16()) {
      SPDLOG_INFO("platformHasFastFp16. FP16 will be used");
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
      use_only_fp32 = false;
    }

    if (int8_enable.count(precision.precision)) {
      if (!builder->platformHasFastInt8()) {
        SPDLOG_ERROR("platform does not Have Fast Int8");
        return nullptr;
      } else {
        SPDLOG_INFO("platformHasFastInt8. Int8 will be used");
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        use_only_fp32 = false;
      }
    }

    modify_layers_precision(precision.precision_fp32, network.get(), nvinfer1::DataType::kFLOAT);
    modify_layers_precision(precision.precision_fp16, network.get(), nvinfer1::DataType::kHALF);
    modify_layers_precision(precision.precision_output_fp32, network.get(),
                            nvinfer1::DataType::kFLOAT, true);
    modify_layers_precision(precision.precision_output_fp16, network.get(),
                            nvinfer1::DataType::kHALF, true);
    if (!use_only_fp32) parse_ln(network.get());
#if (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 5)
    config->setPreviewFeature(nvinfer1::PreviewFeature::kFASTER_DYNAMIC_SHAPES_0805, true);
    SPDLOG_INFO("use tensorrt's PreviewFeature: kFASTER_DYNAMIC_SHAPES_0805");
#endif
#if NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 6
    config->setPreviewFeature(nvinfer1::PreviewFeature::kPROFILE_SHARING_0806, true);
    SPDLOG_INFO("use tensorrt's PreviewFeature: kPROFILE_SHARING_0806");
#endif

    if (!input_reorder.empty()) {
      IPIPE_ASSERT(input_reorder.size() == network->getNbInputs());
    } else {
      // [DEAFULT] set input_reorder to n_inputs-1, ... , 0
      input_reorder.resize(network->getNbInputs());
      // std::iota(input_reorder.rbegin(), input_reorder.rend(), 0);
      std::iota(input_reorder.begin(), input_reorder.end(), 0);
    }

    std::vector<std::pair<std::string, nvinfer1::Dims>> net_inputs_ordered_dims;
    // std::vector<uint32_t> new_location;
    for (int i = 0; i < input_reorder.size(); ++i) {
      net_inputs_ordered_dims.push_back({network->getInput(input_reorder[i])->getName(),
                                         network->getInput(input_reorder[i])->getDimensions()});
    }
    // for (int i = 0; i < network->getNbInputs(); ++i) {
    //   new_location.push_back(
    //       std::distance(net_inputs_ordered_dims.begin(),
    //                     net_inputs_ordered_dims.find(network->getInput(i)->getName())));
    // }

    //// 打印网络输入输出形状
    std::stringstream ss;
    ss << "Network input: ";
    // int index = 0;
    int ch = 0;
    for (std::size_t i = 0; i < net_inputs_ordered_dims.size(); ++i) {
      const auto& item = net_inputs_ordered_dims[i];
      ss << "\n" << input_reorder[i] << ". " << item.first << " ";
      for (int j = 0; j < item.second.nbDims; ++j) {
        const int inputS = item.second.d[j];
        if (j == 1) {
          ch = inputS;
        }
        ss << inputS;
        if (j != item.second.nbDims - 1) ss << "x";
      }
      ss << " ";
    }

    std::string current_order = "(current: ";
    for (std::size_t i = 0; i < input_reorder.size(); ++i) {
      current_order += std::to_string(input_reorder[i]) + ",";
    }
    current_order.pop_back();
    current_order += ")";

    SPDLOG_INFO(colored("use this information to set ranges(batchsizes) of profiles: \n" +
                        ss.str() + "\nreset order by setting `input_reorder`." + current_order));

    nvinfer1::ITensor* input = network->getInput(0);

    if (!_mean.empty() || !_std.empty()) {
      if (net_inputs_ordered_dims.size() != 1) {
        throw std::invalid_argument("there must be only one input if mean or std is setted");
      }
      if (ch != 3) {
        throw std::invalid_argument("channel must be euqal to 3 if mean or std is setted");
      }
      SPDLOG_WARN("start merging mean/std into the network.");
      std::set<nvinfer1::ILayer*> new_layers;
      // modified from
      // https://github.com/wang-xinyu/tensorrtx/blob/d9bdd7e59f19fe1fcc33de64e61ab54345f3e31c/ibnnet/layers.cpp

      nvinfer1::ITensor* pre_input = MeanStd(
          network.get(), input, _mean.empty() ? nullptr : _mean.data(),
          _std.empty() ? nullptr : _std.data(), new_layers, int8_enable.count(precision.precision));

      // pre_input->setPrecision(nvinfer1::DataType:: kFLOAT);

      for (auto i = 0; i < network->getNbLayers(); ++i) {
        auto* layer = network->getLayer(i);
        if (new_layers.find(layer) != new_layers.end()) continue;
        for (auto j = 0; j < layer->getNbInputs(); ++j) {
          if (layer->getInput(j) == input) {
            layer->setInput(j, *pre_input);  // 改变原有第一层的输入
            break;
          }
        }
      }
    }

    auto profile_num = mins.size();
    nvinfer1::IOptimizationProfile* first_profile = nullptr;
    for (size_t index_p = 0; index_p < profile_num; ++index_p) {
      auto profile = builder->createOptimizationProfile();
      if (!first_profile) first_profile = profile;
      if (mins[index_p].empty() || maxs[index_p].empty()) {
        SPDLOG_ERROR("mins[index_p].empty() || maxs[index_p].empty() ");
        return nullptr;
      }

      if (mins[index_p].size() < network->getNbInputs()) {
        mins[index_p].resize(network->getNbInputs(), mins[index_p].back());
      }
      if (maxs[index_p].size() < network->getNbInputs()) {
        maxs[index_p].resize(network->getNbInputs(), maxs[index_p].back());
      }

      for (int i = 0; i < input_reorder.size(); ++i) {
        if (network->getInput(input_reorder[i])->isShapeTensor()) {
          // auto min_dim = shape_tensor_to_dim(mins[index_p][i]);
          // auto max_dim = shape_tensor_to_dim(maxs[index_p][i]);
          auto& min_dim = mins[index_p][i];
          auto& max_dim = maxs[index_p][i];
          // IPIPE_ASSERT(min_dim.nbDims == max_dim.nbDims);

          profile->setShapeValues(network->getInput(input_reorder[i])->getName(),
                                  nvinfer1::OptProfileSelector::kMIN, min_dim.data(),
                                  min_dim.size());
          profile->setShapeValues(network->getInput(input_reorder[i])->getName(),
                                  nvinfer1::OptProfileSelector::kMAX, max_dim.data(),
                                  max_dim.size());
          profile->setShapeValues(network->getInput(input_reorder[i])->getName(),
                                  nvinfer1::OptProfileSelector::kOPT, max_dim.data(),
                                  max_dim.size());

          continue;
        }
        auto net_shape = network->getInput(input_reorder[i])->getDimensions();
        auto min_dim = vtodim(mins[index_p][i], net_shape);
        auto max_dim = vtodim(maxs[index_p][i], net_shape);
        profile->setDimensions(network->getInput(input_reorder[i])->getName(),
                               nvinfer1::OptProfileSelector::kMIN, min_dim);
        profile->setDimensions(network->getInput(input_reorder[i])->getName(),
                               nvinfer1::OptProfileSelector::kOPT, max_dim);
        profile->setDimensions(network->getInput(input_reorder[i])->getName(),
                               nvinfer1::OptProfileSelector::kMAX, max_dim);
        if (!(max_dim.nbDims > 0 && min_dim.nbDims > 0)) {
          SPDLOG_ERROR(
              "max_dim.nbDims = {} net_shape.nbDims={} check failed: max_dim.nbDims > 0 && "
              "min_dim.nbDims > 0",
              max_dim.nbDims, min_dim.nbDims, net_shape.nbDims);
          throw std::invalid_argument("max_dim.nbDims > 0 && min_dim.nbDims > 0");
        }

        if (max_dim.d[0] > min_dim.d[0]) {
          if (!check_dynamic_batchsize(network.get())) {
            auto net_out_shape = network->getOutput(0)->getDimensions();
            IPIPE_ASSERT(net_out_shape.nbDims > 0);
            if (net_out_shape.d[0] != -1) {
              std::stringstream ss;
              ss << "The 0th dimension of the shape for network output should be the batch "
                    "dimension and it needs to be dynamic when input is. However, net_out_shape=";
              ss << net_out_shape.d[0];
              for (auto in = 1; in < net_out_shape.nbDims; ++in) {
                ss << "x" << net_out_shape.d[in];
              }
              ss << ".";

              throw std::invalid_argument(ss.str());
            }
          }
        }
      }
      config->addOptimizationProfile(profile);
    }

    std::unique_ptr<Calibrator> calibrationStream = std::unique_ptr<Calibrator>(new Calibrator());
    if (int8_enable.count(precision.precision) && !is_qat(network.get())) {
      auto dims = first_profile->getDimensions(network->getInput(0)->getName(),
                                               nvinfer1::OptProfileSelector::kOPT);
      if (dims.nbDims == -1) {
        SPDLOG_ERROR("error first_profile");
        return nullptr;
      }

      auto profileCalib = builder->createOptimizationProfile();

      profileCalib->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
      profileCalib->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
      profileCalib->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
      config->setCalibrationProfile(profileCalib);
      // config->setProfileStream(c10::cuda::getCurrentCUDAStream());
      if (true) {
        // config->setFlag(nvinfer1::BuilderFlag::kINT8);
        std::unordered_map<std::string, std::string> calibrationStream_config = config_param;
        calibrationStream_config["calibrate_batchsize"] = std::to_string(dims.d[0]);

        if (!calibrationStream->init(calibrationStream_config, nullptr)) {
          SPDLOG_ERROR("Calibrator init failed");
          config->setInt8Calibrator(nullptr);
          // return nullptr;
        } else
          config->setInt8Calibrator(calibrationStream.get());
      }
    }

    SPDLOG_INFO("Build engine with {} profiles and precision={}...(this can take some time)",
                profile_num, precision.precision);
    auto time_now = now();

    unique_ptr_destroy<nvinfer1::IHostMemory> p_engine_plan(
        builder->buildSerializedNetwork(*network, *config));

    auto time_pass = time_passed(time_now);
    SPDLOG_INFO("finish building engine within {} seconds", int(time_pass / 1000.0));

    if (p_engine_plan) {
      engine_plan = std::string((char*)p_engine_plan->data(), p_engine_plan->size());
      std::shared_ptr<CudaEngineWithRuntime> en_with_rt = loadEngineFromBuffer(engine_plan);

      SPDLOG_INFO("Building engine finished. size of engine is {} MB",
                  int(100 * engine_plan.size() / (1024 * 1024)) / 100.0);

#if NV_TENSORRT_MAJOR >= 8
      if (!precision.timecache.empty()) {
        // if (!Is_File_Exist(precision.timecache))
        { writeTimeCache(precision.timecache, config.get(), time_cache.get()); }
      }
#endif

      return en_with_rt;
    } else {
      return nullptr;
    }
  } else {
    SPDLOG_ERROR("could not parse input engine.");
  }
  return nullptr;
}

}  // namespace ipipe
#endif