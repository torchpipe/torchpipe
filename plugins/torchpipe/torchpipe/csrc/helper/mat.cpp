#include "helper/mat.hpp"
// #include <torch/torch.h>
#include "helper/torch.hpp"

namespace torchpipe::convert{
ImageData TorchAny2ImageData(om::any tensor) {
  torch::Tensor data = tensor.cast<torch::Tensor>();
  return torchpipe::torch2ImageData(data);
}

om::any imageDataToAnyTorchCPU(const convert::ImageData& img) {
  return torchpipe::imageDataToTorchCPU(img);
}

om::any imageDataToAnyTorchGPU(const convert::ImageData& img) {
  return torchpipe::imageDataToTorchCPU(img).cuda();
}
}