#include "torchplugins/CropTensor.hpp"
#include "helper/task_keys.hpp"
#include "helper/torch.hpp"

using namespace omniback;

namespace torchpipe {

namespace {

// https://pytorch.org/cppdocs/notes/tensor_indexing.html
torch::Tensor libtorch_crop(
    torch::Tensor input,
    int x1,
    int y1,
    int x2,
    int y2) {
  if (input.sizes().size() >= 2) { //..hw
    return input.index(
        {"...",
         torch::indexing::Slice(y1, y2),
         torch::indexing::Slice(x1, x2)});
  } else {
    std::stringstream ss;
    ss << "input.sizes() = " << input.sizes() << " x1 y1 x2 y2 = " << x1 << " "
       << y1 << " " << x2 << " " << y2;
    throw std::invalid_argument(ss.str());
  }
}
} // namespace

void CropTensor::forward(const dict& input_dict) {
  auto& input = *input_dict;

  std::vector<int> pbox = dict_get<std::vector<int>>(input_dict, TASK_BOX_KEY);

  auto input_tensor = dict_get<torch::Tensor>(input_dict, TASK_DATA_KEY);

  input_tensor = img_1chw_guard(input_tensor);

  if (pbox.size() < 4) {
    SPDLOG_ERROR("TASK_BOX_KEY: boxes[i].size() < 4");
    throw std::invalid_argument("get an error box");
  }

  auto cropped =
      libtorch_crop(input_tensor, pbox[0], pbox[1], pbox[2], pbox[3]);
  if (cropped.numel() <= 0) {
    SPDLOG_ERROR("get an empty tensor");
    throw std::runtime_error("get an empty tensor");
  }

  input[TASK_RESULT_KEY] = cropped;
}

OMNI_REGISTER(Backend, CropTensor, "CropTensor");

} // namespace torchpipe