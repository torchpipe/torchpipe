#pragma once
#include <set>
#include <string>

#include "NvInfer.h"
#include <NvInferRuntime.h>

namespace torchpipe {
constexprt auto TASK_ENGINE_KEY = "_engine";

void modify_layers_precision(std::set<std::string> precision_fpx,
                             nvinfer1::INetworkDefinition* network,
                             nvinfer1::DataType dataType, bool is_output);
void force_layernorn_fp32(nvinfer1::INetworkDefinition* network);
void print_colored_net(
    nvinfer1::INetworkDefinition* network,
    const std::vector<int>& input_reorder,
    const std::vector<std::pair<std::string, nvinfer1::Dims>>&
        net_inputs_ordered_dims);
void print_net(nvinfer1::INetworkDefinition* network,
               const std::vector<int>& input_reorder,
               const std::vector<std::pair<std::string, nvinfer1::Dims>>&
                   net_inputs_ordered_dims);
void merge_mean_std(nvinfer1::INetworkDefinition* network,
                    const std::vector<float>& mean,
                    const std::vector<float>& std);
bool initTrtPlugins();
nvinfer1::Dims infer_shape(std::vector<int> config_shape,
                           const nvinfer1::Dims& net_input);

struct OnnxParams {
    std::string model_path;
    std::string cache_path;
    std::string precision{"fp16"};
    std::set<std::string> precision_fp32;
    std::set<std::string> precision_fp16;
    std::string timecache;
    size_t max_workspace_size{2048};
    std::string log_level{"info"};
    int force_layer_norm_pattern_fp32{0};
    std::vector<float> mean;
    std::vector<float> std;
    std::vector<std::vector<std::vector<int>>>
        mins;  // multiple profiles - multiple inputs - multiDims
    std::vector<std::vector<std::vector<int>>> maxs
};

std::unique_ptr<nvinfer1::IHostMemory> onnx2trt(const OnnxParams& params);

OnnxParams config2params(
    const std::unordered_map<std::string, std::string>& config);
}  // namespace torchpipe