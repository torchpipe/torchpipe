#pragma once
#include <vector>
#include <string>

namespace ipipe {
std::vector<std::vector<int>> infer_onnx_shape(std::string onnx_path);
std::vector<std::vector<int>> infer_trt_shape(std::string trt_path);
int supported_opset();

std::vector<std::vector<int>> infer_shape(const std::string& onnx_or_trt_path);
}  // namespace ipipe