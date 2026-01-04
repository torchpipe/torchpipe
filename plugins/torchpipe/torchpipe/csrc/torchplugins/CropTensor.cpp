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

torch::Tensor libtorch_copy_make_border(
    torch::Tensor input,
    int top,
    int bottom,
    int left,
    int right) {
  if (input.sizes().size() >= 2) { //..hw
    // 使用 torch::nn::functional::pad 进行边界填充
    // 注意：PyTorch 的 pad 参数顺序是 {left, right, top, bottom}
    std::vector<int64_t> pad = {left, right, top, bottom};

    return torch::nn::functional::pad(
        input,
        torch::nn::functional::PadFuncOptions(pad)
            .mode(torch::kConstant)
            .value(0));
  } else {
    std::stringstream ss;
    ss << "input.sizes() = " << input.sizes()
       << " top bottom left right = " << top << " " << bottom << " " << left
       << " " << right;
    throw std::invalid_argument(ss.str());
  }
}

// std::vector<float> getAffineTransform(int src_h, int src_w, int dst_h, int dst_w)
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

void CopyMakeBorderTensor::forward(const dict& input_dict) {
  auto& input = *input_dict;

  // 保留原有的成员变量赋值
  top_ = dict_get<int>(input_dict, "top");
  bottom_ = dict_get<int>(input_dict, "bottom");
  left_ = dict_get<int>(input_dict, "left");
  right_ = dict_get<int>(input_dict, "right");

  auto input_tensor = dict_get<torch::Tensor>(input_dict, TASK_DATA_KEY);

  // 确保输入张量是 1CHW 格式
  input_tensor = img_1chw_guard(input_tensor);

  // 参数检查
  if (top_ < 0 || bottom_ < 0 || left_ < 0 || right_ < 0) {
    std::stringstream ss;
    ss << "CopyMakeBorderTensor: negative padding values not allowed: "
       << "top=" << top_ << " bottom=" << bottom_ << " left=" << left_
       << " right=" << right_;
    SPDLOG_ERROR(ss.str());
    throw std::invalid_argument(ss.str());
  }

  auto padded =
      libtorch_copy_make_border(input_tensor, top_, bottom_, left_, right_);

  if (padded.numel() <= 0) {
    SPDLOG_ERROR("CopyMakeBorderTensor: get an empty tensor after padding");
    throw std::runtime_error("get an empty tensor after padding");
  }

  input[TASK_RESULT_KEY] = padded;
}

OMNI_REGISTER(Backend, CopyMakeBorderTensor, "CopyMakeBorderTensor");

// void WarpAffineTensor::impl_init(
//     const std::unordered_map<std::string, std::string>& config,
//     const dict& kwargs) {
//   target_h_ = omniback::str::str2int<int>(config, "target_h");
//   target_h_ = omniback::str::str2int<int>(config, "target_w");

//   // 参数检查
//   if (target_h_ > 1024 * 1024 || target_w_ > 1024 * 1024 || target_h_ <= 0 ||
//       target_w_ <= 0 || target_w_ * (target_h_ / 1024.0) > 1024.0 * 1024) {
//     SPDLOG_ERROR(
//         "WarpAffineTensor: illegal h or w: h=" + std::to_string(target_h_) +
//         " w=" + std::to_string(target_w_));
//     throw std::invalid_argument("WarpAffineTensor: illegal h or w");
//   }
// }
// void WarpAffineTensor::forward(const dict& input_dict) 
// {

// }
// OMNI_REGISTER(Backend, WarpAffineTensor, "WarpAffineTensor");

} // namespace torchpipe