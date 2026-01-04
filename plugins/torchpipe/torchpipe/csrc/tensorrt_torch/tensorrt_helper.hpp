#pragma once
#include <set>
#include <string>

#include "NvInfer.h"
#include <NvInferRuntime.h>
#include "helper/net_info.hpp"

namespace torchpipe {

nvinfer1::DataType convert2trt(const std::string& type_name);

inline NetIOInfo::Dims64 convert_dims(const nvinfer1::Dims& dims) {
  NetIOInfo::Dims64 dims64;
  dims64.nbDims = dims.nbDims;
  for (int i = 0; i < dims.nbDims; ++i) {
    dims64.d[i] = static_cast<int64_t>(dims.d[i]);
  }
  return dims64;
}

inline nvinfer1::Dims convert_dims(const NetIOInfo::Dims64& dims64) {
  nvinfer1::Dims dims;
  dims.nbDims = dims64.nbDims;
  for (int i = 0; i < dims64.nbDims; ++i) {
    dims.d[i] = dims64.d[i];
  }
  return dims;
}

constexpr auto TASK_ENGINE_KEY = "_engine";

void modify_layers_precision(
    std::set<std::string> precision_fpx,
    nvinfer1::INetworkDefinition* network,
    nvinfer1::DataType dataType,
    bool is_output);
void force_layernorn_fp32(nvinfer1::INetworkDefinition* network);
void print_colored_net(
    nvinfer1::INetworkDefinition* network,
    const std::vector<int>& input_reorder,
    const std::vector<std::pair<std::string, nvinfer1::Dims>>&
        net_inputs_ordered_dims);
void print_net(
    nvinfer1::INetworkDefinition* network,
    const std::vector<int>& input_reorder,
    const std::vector<std::pair<std::string, nvinfer1::Dims>>&
        net_inputs_ordered_dims);
void merge_mean_std(
    nvinfer1::INetworkDefinition* network,
    const std::vector<float>& mean,
    const std::vector<float>& std);
// void add_anchor_plugins(
//     nvinfer1::INetworkDefinition* network,
//     const std::vector<std::string>& names,
//     bool with_pre = true,
//     bool with_post = true);

bool initTrtPlugins();
nvinfer1::Dims infer_shape(
    std::vector<int> config_shape,
    const nvinfer1::Dims& net_input);

struct OnnxParams {
  std::string model; // moddel
  std::string model_cache; // model::cache
  std::string precision{"fp16"};
  std::set<std::string> precision_fp32; // precision::fp32
  std::set<std::string> precision_fp16; // precision::fp16
  std::string timingcache; // model::timingcache
  size_t max_workspace_size{2048};
  std::string log_level{"info"};
  int force_layer_norm_pattern_fp32{1};
  std::vector<float> mean;
  std::vector<float> std;
  std::vector<std::vector<std::vector<int>>>
      mins; // min: multiple profiles - multiple inputs - multiDims
  std::vector<std::vector<std::vector<int>>> maxs; // max
  size_t instance_num = 1;
  std::string hardward_compatibility = "NONE"; // = "AMPERE_PLUS";
};

std::unique_ptr<nvinfer1::IHostMemory> onnx2trt(OnnxParams& params);

OnnxParams config2onnxparams(
    const std::unordered_map<std::string, std::string>& config);

nvinfer1::ILogger* get_trt_logger();

NetIOInfos get_context_shape(
    nvinfer1::IExecutionContext* context,
    size_t profile_index);

// MultiProfileNetIOInfos create_contexts(nvinfer1::ICudaEngine* engine);
// std::unique_ptr<nvinfer1::IExecutionContext> create_context(NetIOInfos&
// info);

std::unique_ptr<nvinfer1::IExecutionContext> create_context(
    nvinfer1::ICudaEngine* engine,
    size_t instance_index);

} // namespace torchpipe

#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 2) || \
    (NV_TENSORRT_MAJOR >= 11)
#define TRT_USER_MANAGED_MEM 1
#else
#define TRT_USER_MANAGED_MEM 0
#endif
