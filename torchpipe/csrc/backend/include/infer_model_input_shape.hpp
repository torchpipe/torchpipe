#pragma once
#include <vector>
#include <string>

namespace ipipe {
std::vector<std::vector<int>> infer_onnx_shape(std::string onnx_path);
std::vector<std::vector<int>> infer_trt_shape(std::string trt_path);

}  // namespace ipipe