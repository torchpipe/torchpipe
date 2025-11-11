
// #include <torch/serialize.h>

#include "torchplugins/CvtColorTensor.hpp"
#include "helper/task_keys.hpp"
#include "helper/torch.hpp"

using namespace omniback;

namespace torchpipe {

void CvtColorTensor::impl_init(
    const std::unordered_map<std::string, std::string>& config_param,
    const dict& kwargs) {
  auto iter = config_param.find(TASK_COLOR_KEY);
  OMNI_ASSERT(iter != config_param.end(), "CvtColorTensor: color is not set");

  color_ = iter->second;
  OMNI_ASSERT(
      VALID_COLOR_SPACE.count(color_) != 0,
      "color must be rgb or bgr: " + color_);
}

void CvtColorTensor::forward(const dict& input_dict) {
  auto src_color = input_dict->find(TASK_COLOR_KEY);
  OMNI_ASSERT(src_color != input_dict->end(), "input dict must contain color");

  std::string input_color = any_cast<std::string>(src_color->second);

  if (input_color == color_) {
    (*input_dict)[TASK_RESULT_KEY] = (*input_dict)[TASK_DATA_KEY];
    return;
  }
  OMNI_ASSERT(
      VALID_COLOR_SPACE.count(input_color) != 0,
      input_color + " is not supported yet");

#if 1
  auto input_tensor =
      omniback::dict_get<torch::Tensor>(input_dict, TASK_DATA_KEY);

  const auto mem = guard_valid_memory_format(input_tensor);
  switch (mem) {
    case MemoryFormat::NCHW: {
      input_tensor = torch::flip(input_tensor, {1});
      break;
    }
    case MemoryFormat::HWC: {
      input_tensor = torch::flip(input_tensor, {2});
      break;
    }
  }

  (*input_dict)[TASK_COLOR_KEY] = color_;
  (*input_dict)[TASK_RESULT_KEY] = input_tensor;
#endif
}

OMNI_REGISTER(Backend, CvtColorTensor, "cvtColorTensor,CvtColorTensor");

} // namespace torchpipe