#ifndef TENSORRT_PLUGIN_ANCHOR_PLUGIN_H
#define TENSORRT_PLUGIN_ANCHOR_PLUGIN_H

#include <NvInferRuntime.h>
#if NV_TENSORRT_MAJOR >= 10
#include <NvInferRuntimePlugin.h>
#endif
#include <string>
#include <unordered_map>
#include <vector>

// In IPluginV3 interface, the plugin name, version, and name space must be
// specified for the plugin and plugin creator exactly the same.
constexpr char const* const kANCHOR_PLUGIN_NAME{"AnchorPlugin"};
constexpr char const* const kANCHOR_PLUGIN_VERSION{"1"};
constexpr char const* const kANCHOR_PLUGIN_NAMESPACE{""};
namespace omniback {
class Backend;
}
namespace nvinfer1 {
namespace plugin {

struct AnchorPluginParameters {
  int32_t num_output{1};
  int32_t num_input{1};
  int32_t layer_idx{0};
  size_t workspace_size{0};
  std::vector<nvinfer1::DataType> type;
  std::string name{"AnchorPlugin"};
};

#if (NV_TENSORRT_MAJOR >= 10 && NV_TENSORRT_MINOR >= 3) || \
    NV_TENSORRT_MAJOR >= 11
class AnchorPlugin : public IPluginV3,
                     public IPluginV3OneCore,
                     public v_2_0::IPluginV3OneBuild,
                     public IPluginV3OneRuntime {

#elif NV_TENSORRT_MAJOR >= 10
class AnchorPlugin : public IPluginV3,
                     public IPluginV3OneCore,
                     public IPluginV3OneBuild,
                     public IPluginV3OneRuntime {
#endif

#if NV_TENSORRT_MAJOR >= 10
public:
AnchorPlugin(std::string const& params, bool is_build_phase = false);

~AnchorPlugin() override {
  delete dependency_;
  dependency_ = nullptr;
  }

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

#if (NV_TENSORRT_MAJOR >= 10 && NV_TENSORRT_MINOR >= 3) || \
    NV_TENSORRT_MAJOR >= 11
  int32_t getAliasedInput(int32_t outputIndex) noexcept override {
    return outputIndex;
  }
#endif

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
  AnchorPluginParameters anchor_params_;
  std::string serialization_;

  void initFieldsToSerialize();

  std::vector<nvinfer1::PluginField> mDataToSerialize;
  nvinfer1::PluginFieldCollection mFCToSerialize;

  omniback::Backend* dependency_{nullptr};

  bool is_build_phase_{false};
};
#endif
} // namespace plugin
} // namespace nvinfer1

namespace nvinfer1 {
namespace plugin {

#if NV_TENSORRT_MAJOR >= 10
// Plugin factory class.
class AnchorPluginCreator : public nvinfer1::IPluginCreatorV3One {
 public:
  AnchorPluginCreator();

  ~AnchorPluginCreator() override = default;

  nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

  IPluginV3* createPlugin(
      char const* name,
      PluginFieldCollection const* fc,
      TensorRTPhase phase) noexcept override;

 public:
  // BaseCreator
  char const* getPluginNamespace() const noexcept override {
    return kANCHOR_PLUGIN_NAMESPACE;
  }

  char const* getPluginName() const noexcept override {
    return kANCHOR_PLUGIN_NAME;
  }

  char const* getPluginVersion() const noexcept override {
    return kANCHOR_PLUGIN_VERSION;
  }

 private:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
};

#endif
} // namespace plugin
} // namespace nvinfer1
#endif // TENSORRT_PLUGIN_ANCHOR_PLUGIN_H