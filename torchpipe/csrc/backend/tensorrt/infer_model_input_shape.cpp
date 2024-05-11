#include "infer_model_input_shape.hpp"

#include "dynamic_onnx2trt.hpp"
#include "tensorrt_utils.hpp"

#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif
#include "params.hpp"
namespace ipipe {

#if NV_TENSORRT_MAJOR < 10
std::vector<std::vector<int>> infer_shape(std::shared_ptr<CudaEngineWithRuntime> engine) {
  const unsigned n_profiles = engine->engine->getNbOptimizationProfiles();
  const unsigned n_inputsOutputs = engine->engine->getNbIOTensors();

  constexpr auto profile_index = 0;

  std::vector<std::vector<int>> ret;
  for (unsigned i = profile_index; i < profile_index + 1; i++) {
    for (unsigned j = 0; j < n_inputsOutputs; j++) {
      const auto index = i * n_inputsOutputs + j;

      if (engine->engine->bindingIsInput(index)) {
        // 获取tensorrt引擎的输入维度
        const auto dims = engine->engine->getBindingDimensions(index);
        std::vector<int> shape;
        for (int k = 0; k < dims.nbDims; k++) {
          shape.push_back(dims.d[k]);
        }
        ret.push_back(shape);
      }
    }
  }

  return ret;
}
#else
std::vector<std::vector<int>> infer_shape(std::shared_ptr<CudaEngineWithRuntime> engine) {
  const unsigned n_profiles = engine->engine->getNbOptimizationProfiles();
  const unsigned n_inputsOutputs = engine->engine->getNbIOTensors();

  constexpr auto profile_index = 0;

  std::vector<std::vector<int>> ret;
  for (unsigned i = 0; i < n_profiles; i++) {
    for (unsigned j = 0; j < n_inputsOutputs; j++) {
      const auto index = i * n_inputsOutputs + j;
      const auto name = engine->engine->getIOTensorName(j);
      const auto tensorType = engine->engine->getTensorIOMode(name);

      if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
        // 获取tensorrt引擎的输入维度
        // const auto dims = engine->engine->getBindingDimensions(index);
        nvinfer1::Dims dims = engine->engine->getTensorShape(name);
        std::vector<int> shape;
        for (int k = 0; k < dims.nbDims; k++) {
          shape.push_back(dims.d[k]);
        }
        ret.push_back(shape);
      }
    }
  }

  return ret;
}
#endif

std::vector<std::vector<int>> infer_trt_shape(std::string trt_path) {
  std::string engine_plan;
  auto engine = loadCudaBackend(trt_path, ".trt", engine_plan);
  return infer_shape(engine);
}

std::vector<std::vector<int>> infer_shape(const std::string& onnx_or_trt_path) {
  if (endswith(onnx_or_trt_path, ".onnx")) {
    return infer_onnx_shape(onnx_or_trt_path);
  } else if (endswith(onnx_or_trt_path, ".trt")) {
    return infer_trt_shape(onnx_or_trt_path);
  } else {
    throw std::runtime_error("onnx_or_trt_path must be endswith .onnx or .trt");
  }
}

int supported_opset() {
  if ((NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >= 6) || (NV_TENSORRT_MAJOR >= 9))
    return 17;
  else
    return 13;
};

#ifdef PYBIND
#ifdef WITH_TENSORRT
namespace py = pybind11;
// 通过pybind11将infer_onnx_shape函数暴露到python中，注意其参数
void init_infer_shape(py::module& m) {
  m.def("infer_shape", py::overload_cast<const std::string&>(&infer_shape),
        py::call_guard<py::gil_scoped_release>(), py::arg("model_path"));
}
void supported_opset(py::module& m) {
  m.def("supported_opset", py::overload_cast<>(&supported_opset),
        py::call_guard<py::gil_scoped_release>());
}
#endif
#endif
}  // namespace ipipe