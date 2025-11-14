#ifndef TENSORRT_PLUGIN_PLUGIN_H
#define TENSORRT_PLUGIN_PLUGIN_H

#include <string>
#include <vector>

#include <NvInferRuntime.h>
#if NV_TENSORRT_MAJOR >= 10
#include <NvInferRuntimePlugin.h>
#endif
#include "omniback/core/backend.hpp"

// In IPluginV3 interface, the plugin name, version, and name space must be
// specified for the plugin and plugin creator exactly the same.
constexpr char const* const kTORCH_PLUGIN_NAME{"TorchPlugin"};
constexpr char const* const kTORCH_PLUGIN_VERSION{"1"};
constexpr char const* const kTORCH_PLUGIN_NAMESPACE{""};
// namespace omniback {
// class Backend;
// }
namespace nvinfer1 {
namespace plugin {

struct TorchPluginParameters {
  // int32_t group;
  // nvinfer1::DataType dtype;
  // int32_t channelSize;
  // int32_t height;
  // int32_t width;
  // size_t dtypeBytes;

  // size_t index{0};
  int32_t num_output{1};
  int32_t num_input{1};
  int32_t layer_idx{0};
  size_t workspace_size{0};
  std::vector<nvinfer1::DataType> type;
  std::string name{"TorchPlugin"};
};

#if NV_TENSORRT_MAJOR >= 10
class TorchPlugin : public IPluginV3,
                    public IPluginV3OneCore,
                    public IPluginV3OneBuild,
                    public IPluginV3OneRuntime {
 public:
  TorchPlugin(std::string const& params, bool is_build_phase = false);

  ~TorchPlugin() override = default;

  // IPluginV3 Methods

  IPluginCapability* getCapabilityInterface(
      PluginCapabilityType type) noexcept override;

  IPluginV3* clone() noexcept override;

  // IPluginV3OneCore Methods

  char const* getPluginName() const noexcept override;

  char const* getPluginVersion() const noexcept override;

  char const* getPluginNamespace() const noexcept override;

  // IPluginV3OneBuild Methods

  int32_t getNbOutputs() const noexcept override;

  int32_t configurePlugin(
      DynamicPluginTensorDesc const* in,
      int32_t nbInputs,
      DynamicPluginTensorDesc const* out,
      int32_t nbOutputs) noexcept override;

  bool supportsFormatCombination(
      int32_t pos,
      DynamicPluginTensorDesc const* inOut,
      int32_t nbInputs,
      int32_t nbOutputs) noexcept override;

  int32_t getOutputDataTypes(
      DataType* outputTypes,
      int32_t nbOutputs,
      DataType const* inputTypes,
      int32_t nbInputs) const noexcept override;

  int32_t getOutputShapes(
      DimsExprs const* inputs,
      int32_t nbInputs,
      DimsExprs const* shapeInputs,
      int32_t nbShapeInputs,
      DimsExprs* outputs,
      int32_t nbOutputs,
      IExprBuilder& exprBuilder) noexcept override;

  // IPluginV3OneRuntime Methods

  int32_t enqueue(
      PluginTensorDesc const* inputDesc,
      PluginTensorDesc const* outputDesc,
      void const* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;

  int32_t onShapeChange(
      PluginTensorDesc const* in,
      int32_t nbInputs,
      PluginTensorDesc const* out,
      int32_t nbOutputs) noexcept override;

  IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

  PluginFieldCollection const* getFieldsToSerialize() noexcept override;

  size_t getWorkspaceSize(
      DynamicPluginTensorDesc const* inputs,
      int32_t nbInputs,
      DynamicPluginTensorDesc const* outputs,
      int32_t nbOutputs) const noexcept override;

  //  public:

 private:
  // TensorRT plugin parameters.
  std::unordered_map<std::string, std::string> params_;
  TorchPluginParameters torch_params_;
  std::string serialization_;

  void initFieldsToSerialize();

  std::vector<nvinfer1::PluginField> mDataToSerialize;
  nvinfer1::PluginFieldCollection mFCToSerialize;

  std::unique_ptr<omniback::Backend> dependency_{nullptr};

  bool is_build_phase_{false};

  //  private:
  //   void set_to_build_phase() {
  //     dependency_.release();
  //   }
  // bool is_in_runtime_{false};
};
#endif
} // namespace plugin
} // namespace nvinfer1

namespace nvinfer1 {
namespace plugin {

#if NV_TENSORRT_MAJOR >= 10
// Plugin factory class.
class TorchPluginCreator : public nvinfer1::IPluginCreatorV3One {
 public:
  TorchPluginCreator();

  ~TorchPluginCreator() override = default;

  nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

  IPluginV3* createPlugin(
      char const* name,
      PluginFieldCollection const* fc,
      TensorRTPhase phase) noexcept override;

 public:
  // BaseCreator
  char const* getPluginNamespace() const noexcept override {
    return kTORCH_PLUGIN_NAMESPACE;
  }

  char const* getPluginName() const noexcept override {
    return kTORCH_PLUGIN_NAME;
  }

  char const* getPluginVersion() const noexcept override {
    return kTORCH_PLUGIN_VERSION;
  }

 private:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
};

#endif
} // namespace plugin
} // namespace nvinfer1
#endif // TENSORRT_PLUGIN_PLUGIN_H