#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include "omniback/core/any.hpp"

namespace torchpipe::convert{

struct ImageData {
  void* data = nullptr;
  size_t rows = 0;
  size_t cols = 0;
  size_t channels = 0;
  bool is_float = false; // true: float32, false: uint8
  std::function<void(void*)> deleter;
};
ImageData TorchAny2ImageData(om::any tensor);

om::any imageDataToAnyTorchCPU(const convert::ImageData& img);

om::any imageDataToAnyTorchGPU(const convert::ImageData& img);
}