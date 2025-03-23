#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include "hami/helper/base_logging.hpp"
#include "hami/helper/macro.h"
#include "hami/helper/string.hpp"

#include "hami/core/reflect.h"

#include "tensorrt_torch/tensorrt_plugins.hpp"

#include <c10/cuda/CUDAStream.h> // 必须包含此头文件
#include <torch/torch.h>
#include "c10/cuda/CUDAGuard.h"

namespace nvinfer1 {
namespace plugin {

TorchPlugin::TorchPlugin(const std::string& params) : serialization_(params) {
  mParams = hami::str::map_split(params, '=', ',');
  //       HAMI_ASSERT(
  //         mParams.find("num_output") != mParams.end,
  //         "No `num_output` find in params. Export onnx with `params` attr.
  //         and `num_output=*`"); // abort
  //     torch_params_.num_output = std::soti(mParams["num_output"]);
  // hami::str::try_update(mParams, "index", torch_params_.index);
  hami::str::try_update(mParams, "num_output", torch_params_.num_output);

  hami::str::try_update<std::string>(mParams, "name", torch_params_.name);

  initFieldsToSerialize();

  [[maybe_unused]] static auto _ = [this]() {
    SPDLOG_INFO("name={}", torch_params_.name);
    return 1;
  }();

  dependency_ = HAMI_INSTANCE_GET(hami::Backend, torch_params_.name);
  HAMI_ASSERT(dependency_);
}

void TorchPlugin::initFieldsToSerialize() {
  // Serialize TorchPluginParameters.
  mDataToSerialize.clear();
  mDataToSerialize.emplace_back(nvinfer1::PluginField(
      "params",
      serialization_.data(),
      PluginFieldType::kCHAR,
      serialization_.size()));
  mFCToSerialize.nbFields = mDataToSerialize.size();
  mFCToSerialize.fields = mDataToSerialize.data();
}

IPluginCapability* TorchPlugin::getCapabilityInterface(
    PluginCapabilityType type) noexcept {
  try {
    if (type == PluginCapabilityType::kBUILD) {
      return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME) {
      return static_cast<IPluginV3OneRuntime*>(this);
    }
    HAMI_ASSERT(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore*>(this);
  } catch (std::exception const& e) {
    SPDLOG_ERROR("Got exception: {}", e.what());
  }
  return nullptr;
}

IPluginV3* TorchPlugin::clone() noexcept {
  // It's possible to encounter errors during cloning.
  // For example, if the memory to allocate is insufficient, exceptions can be
  // thrown.
  try {
    IPluginV3* const plugin{new TorchPlugin{serialization_}};
    return plugin;
  } catch (std::exception const& e) {
    SPDLOG_ERROR("Got exception: {}", e.what());
  }
  return nullptr;
}

char const* TorchPlugin::getPluginName() const noexcept {
  //   HAMI_ASSERT(
  //       mParams.find("name") != mParams.end,
  //       "No `name` find in params. Export onnx with `params` attr. and
  //       `name=*`"); // abort
  //   return mParams["name"];
  return kTORCH_PLUGIN_NAME;
}

char const* TorchPlugin::getPluginVersion() const noexcept {
  return kTORCH_PLUGIN_VERSION;
}

char const* TorchPlugin::getPluginNamespace() const noexcept {
  return kTORCH_PLUGIN_NAMESPACE;
}

int32_t TorchPlugin::getNbOutputs() const noexcept {
  // SPDLOG_INFO("getNbOutputs called. num_output = {}",
  // torch_params_.num_output);
  return torch_params_.num_output;
  //   return 1;
}

int32_t TorchPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
  // Communicates the number of inputs and outputs, dimensions, and datatypes
  // of all inputs and outputs, broadcast information for all inputs and
  // outputs, the chosen plugin format, and maximum batch size. At this point,
  // the plugin sets up its internal state and selects the most appropriate
  // algorithm and data structures for the given configuration. Note: Resource
  // allocation is not allowed in this API because it causes a resource leak.

  // This member function will only be called during engine build time.

  // Validate input arguments.
  torch_params_.num_input = nbInputs;
  torch_params_.num_output = nbOutputs;

  [[maybe_unused]] static auto _ = [nbInputs, nbOutputs, in, out] {
    SPDLOG_INFO(
        "configurePlugin. nbInputs={}, nbOutputs={}, in[0].desc.dims.nbDims={}, out[0].desc.dims.nbDims={},in[0].desc.type={}",
        nbInputs,
        nbOutputs,
        in[0].desc.dims.nbDims,
        out[0].desc.dims.nbDims,
        int(in[0].desc.type));
    return true;
  }();

  // if (mParams.dtype == nvinfer1::DataType::kINT8) {
  //   mParams.dtypeBytes = 1;
  // } else if (mParams.dtype == nvinfer1::DataType::kHALF) {
  //   mParams.dtypeBytes = 2;
  // } else if (mParams.dtype == nvinfer1::DataType::kFLOAT) {
  //   mParams.dtypeBytes = 4;
  // } else {
  //   HAMI_ASSERT(false);
  // }

  return 0;
}

bool TorchPlugin::supportsFormatCombination(
    int32_t pos,
    DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  // For this method inputs are numbered 0..(nbInputs-1) and outputs are
  // numbered nbInputs..(nbInputs+nbOutputs-1). Using this numbering, pos is
  // an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
  [[maybe_unused]] static auto _ = [nbInputs, nbOutputs, inOut] {
    SPDLOG_INFO(
        "supportsFormatCombination called. nbInputs={}, nbOutputs={}, type={}",
        nbInputs,
        nbOutputs,
        int(inOut[0].desc.type));
    return true;
  }();
  return true;
}

int32_t TorchPlugin::getOutputDataTypes(
    DataType* outputTypes,
    int32_t nbOutputs,
    DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
  // One output.
  SPDLOG_INFO(
      "getOutputDataTypes called. nbInputs={}, nbOutputs={}",
      nbInputs,
      nbOutputs);
  // The output type is the same as the input type.
  for (size_t i = 0; i < nbOutputs; ++i) {
    outputTypes[i] = inputTypes[0];
  }

  return 0;
}

int32_t TorchPlugin::getOutputShapes(
    DimsExprs const* inputs,
    int32_t nbInputs,
    DimsExprs const* shapeInputs,
    int32_t nbShapeInputs,
    DimsExprs* outputs,
    int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept {
  SPDLOG_INFO("getOutputShapes called. ");

  for (size_t i = 0; i < nbOutputs; ++i) {
    outputs[i].nbDims = inputs[0].nbDims;
    for (int32_t j{0}; j < inputs[0].nbDims; ++j) {
      outputs[i].d[j] = inputs[i].d[j];
    }
  }

  return 0;
}

int32_t TorchPlugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  [[maybe_unused]] static auto _ = []() {
    SPDLOG_INFO("ENQUEUE ");
    return true;
  }();

  // 获取当前CUDA设备
  int device_id;
  cudaError_t cuda_status = cudaGetDevice(&device_id);
  if (cuda_status != cudaSuccess) {
    return cuda_status;
  }

  // 2. 条件性创建流守卫
  std::unique_ptr<c10::cuda::CUDAStreamGuard> guard;
  if (c10::cuda::getCurrentCUDAStream(device_id) != stream) {
    guard = std::make_unique<c10::cuda::CUDAStreamGuard>(
        c10::cuda::getStreamFromExternal(stream, device_id));
  }

  //----------------------------------------------
  // 1. 解析输入张量
  //----------------------------------------------
  const int nbInputs = torch_params_.num_input;
  std::vector<torch::Tensor> input_tensors;
  for (int i = 0; i < nbInputs; ++i) { // 使用函数参数nbInputs
    const PluginTensorDesc& desc = inputDesc[i];

    // 数据类型映射
    torch::ScalarType dtype;
    switch (desc.type) {
      case DataType::kFLOAT:
        dtype = torch::kFloat32;
        break;
      case DataType::kHALF:
        dtype = torch::kFloat16;
        break;
      case DataType::kINT32:
        dtype = torch::kInt32;
        break;
      case DataType::kINT64:
        dtype = torch::kInt64;
        break;
      case DataType::kINT8:
        dtype = torch::kInt8;
        break;
      default:
        return cudaErrorInvalidValue;
    }

    // 提取维度
    std::vector<int64_t> sizes;
    for (int d = 0; d < desc.dims.nbDims; ++d) {
      sizes.push_back(desc.dims.d[d]);
    }

    // 计算步长
    std::vector<int64_t> strides(desc.dims.nbDims, 1);
    // 连续内存布局
    for (int d = desc.dims.nbDims - 2; d >= 0; --d) {
      strides[d] = strides[d + 1] * sizes[d + 1];
    }

    // 创建输入张量视图
    input_tensors.emplace_back(torch::from_blob(
        const_cast<void*>(inputs[i]),
        sizes,
        strides,
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id)));
  }

  //----------------------------------------------
  // 2. 构造输出张量容器
  //----------------------------------------------
  const int nbOutputs = torch_params_.num_output;
  std::vector<torch::Tensor> output_tensors;
  for (int o = 0; o < nbOutputs; ++o) { // 使用函数参数nbOutputs
    const PluginTensorDesc& desc = outputDesc[o];

    // 数据类型映射
    torch::ScalarType dtype;
    switch (desc.type) {
      case DataType::kFLOAT:
        dtype = torch::kFloat32;
        break;
      case DataType::kHALF:
        dtype = torch::kFloat16;
        break;
      case DataType::kINT32:
        dtype = torch::kInt32;
        break;
      case DataType::kINT64:
        dtype = torch::kInt64;
        break;
      case DataType::kINT8:
        dtype = torch::kInt8;
        break;
      default:
        return cudaErrorInvalidValue;
    }

    // 提取维度
    std::vector<int64_t> sizes;
    for (int d = 0; d < desc.dims.nbDims; ++d) {
      sizes.push_back(desc.dims.d[d]);
    }

    // 计算步长
    std::vector<int64_t> strides(desc.dims.nbDims, 1);
    // 连续内存布局
    for (int d = desc.dims.nbDims - 2; d >= 0; --d) {
      strides[d] = strides[d + 1] * sizes[d + 1];
    }

    // 创建输出张量视图
    output_tensors.emplace_back(torch::from_blob(
        outputs[o],
        sizes,
        strides,
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id)));
  }

  //----------------------------------------------
  // 3. 调用外部处理函数
  //----------------------------------------------
  try {
    // get_output(output_tensors, input_tensors); // 用户自定义函数
  } catch (const std::exception& e) {
    SPDLOG_ERROR("get_output failed: {}", e.what());
    return cudaErrorUnknown;
  }

  //----------------------------------------------
  // 4. 错误检查（移除冗余事件同步）
  //----------------------------------------------
  return cudaGetLastError();
}

int32_t TorchPlugin::onShapeChange(
    PluginTensorDesc const* in,
    int32_t nbInputs,
    PluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
  return 0;
}

IPluginV3* TorchPlugin::attachToContext(
    IPluginResourceContext* context) noexcept {
  return clone();
}

PluginFieldCollection const* TorchPlugin::getFieldsToSerialize() noexcept {
  return &mFCToSerialize;
}

size_t TorchPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
  return 0;
}

} // namespace plugin
} // namespace nvinfer1

namespace nvinfer1 {
namespace plugin {

// This is not needed for plugin dynamic registration.
REGISTER_TENSORRT_PLUGIN(TorchPluginCreator);

// Plugin creator
TorchPluginCreator::TorchPluginCreator() {
  // Declare the ONNX attributes that the ONNX parser will collect from the
  // ONNX model that contains the TorchPlugin node.

  // In our dummy case,
  // attrs={
  //     "params": parameters,
  // }

  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("params", nullptr, PluginFieldType::kUNKNOWN, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

nvinfer1::PluginFieldCollection const* TorchPluginCreator::
    getFieldNames() noexcept {
  // This is only used in the build phase.
  return &mFC;
}

IPluginV3* TorchPluginCreator::createPlugin(
    char const* name,
    PluginFieldCollection const* fc,
    TensorRTPhase phase) noexcept {
  // The build phase and the deserialization phase are handled differently.
  if (phase == TensorRTPhase::kBUILD) {
    // The attributes from the ONNX node will be parsed and passed via fc.
    try {
      nvinfer1::PluginField const* fields{fc->fields};
      int32_t nbFields{fc->nbFields};

      HAMI_ASSERT(nbFields >= 1, "nbFields = " + std::to_string(nbFields));

      std::string params;
      for (int32_t i{0}; i < nbFields; ++i) {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "params")) {
          SPDLOG_INFO("fields[i].length={}", fields[i].length);
          params.resize(size_t(fields[i].length));
          HAMI_ASSERT(
              fields[i].type == nvinfer1::PluginFieldType::kCHAR,
              std::to_string(int(fields[i].type)));
          memcpy(params.data(), fields[i].data, fields[i].length);

        } else {
          SPDLOG_WARN("Skip unkown `attrName` {}", attrName);
        }
      }
      SPDLOG_INFO("Plugin Attributes: params -> {}", params);
      // check
      // [[maybe_unused]] auto map_params = hami::str::str_split(params, ',');

      TorchPlugin* const plugin{new TorchPlugin{params}};
      return plugin;
    } catch (std::exception const& e) {
      SPDLOG_ERROR("Got exception: {}", e.what());
    }
    return nullptr;
  } else if (phase == TensorRTPhase::kRUNTIME) {
    // The attributes from the serialized plugin will be passed via fc.
    try {
      nvinfer1::PluginField const* fields{fc->fields};
      int32_t nbFields{fc->nbFields};
      HAMI_ASSERT(nbFields == 1);

      char const* attrName = fields[0].name;
      HAMI_ASSERT(!strcmp(attrName, "params"));
      std::string params;
      HAMI_ASSERT(fields[0].type == nvinfer1::PluginFieldType::kCHAR);
      params.resize(size_t(fields[0].length));
      memcpy(params.data(), fields[0].data, fields[0].length);
      TorchPlugin* const plugin{new TorchPlugin{params}};
      return plugin;
    } catch (std::exception const& e) {
      SPDLOG_ERROR("Got exception: {}", e.what());
    }
    return nullptr;
  } else {
    return nullptr;
  }
}

} // namespace plugin
} // namespace nvinfer1
