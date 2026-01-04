#include "torchplugins/ResizeTensor.hpp"
#include "helper/task_keys.hpp"
#include "helper/torch.hpp"
#include <omniback/addons/torch/type_traits.h>

using namespace omniback;

namespace torchpipe {

class ResizeTensor : public BackendOne {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override {
    resize_h_ = omniback::str::str2int<size_t>(config, "resize_h");
    resize_w_ = omniback::str::str2int<size_t>(config, "resize_w");

    if (resize_h_ > 1024 * 1024 || resize_w_ > 1024 * 1024 || resize_h_ <= 1 ||
        resize_w_ <= 1 || resize_w_ * (resize_h_ / 1024.0) > 1024.0 * 1024) {
      SPDLOG_ERROR(
          "ResizeTensor: illigle h or w: h=" + std::to_string(resize_h_) +
          " w=" + std::to_string(resize_w_));
      throw std::invalid_argument("ResizeTensor: illigle h or w");
    }
  }

  /**
   * @brief cpu->gpu
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] =
   * input[TASK_DATA_KEY].cuda()
   */
  virtual void forward(const dict& input_dict) override {
    auto input_tensor = dict_get<torch::Tensor>(input_dict, TASK_DATA_KEY);
    bool is_hwc_tensor = is_hwc(input_tensor);

    input_tensor = img_1chw_guard(input_tensor);
    if (input_tensor.dtype() != torch::kFloat) {
      input_tensor = input_tensor.to(torch::kFloat);
    }

    if (!input_tensor.is_contiguous())
      input_tensor = input_tensor.contiguous();

    at::Tensor im_resize;

    if (input_tensor.size(2) == resize_h_ &&
        input_tensor.size(3) == resize_w_) {
      im_resize = input_tensor;
    } else {
      im_resize = torch::upsample_bilinear2d(
          input_tensor, {(long)resize_h_, (long)resize_w_}, true);
    }
    if (is_hwc_tensor) {
      im_resize = im_resize.permute({0, 2, 3, 1}).squeeze(0);
    }

    (*input_dict)[TASK_RESULT_KEY] = im_resize;
  }

 private:
  size_t resize_h_{0};
  size_t resize_w_{0};
};

OMNI_REGISTER(omniback::Backend, ResizeTensor, "ResizeTensor");

class DynamicResizeTensor : public BackendOne {
 private:
  virtual void forward(const dict& input_dict) override {
    auto input_tensor = dict_get<torch::Tensor>(input_dict, TASK_DATA_KEY);
     resize_h_ =
        dict_get<int>(input_dict, "resize_h");
     resize_w_ = dict_get<int>(input_dict, "resize_w");

     bool is_hwc_tensor = is_hwc(input_tensor);

     input_tensor = img_1chw_guard(input_tensor);
     if (input_tensor.dtype() != torch::kFloat) {
       input_tensor = input_tensor.to(torch::kFloat);
    }

    if (!input_tensor.is_contiguous())
      input_tensor = input_tensor.contiguous();

    torch::Tensor im_resize;

    if (input_tensor.size(2) == resize_h_ &&
        input_tensor.size(3) == resize_w_) {
      im_resize = input_tensor;
    } else {
      im_resize = torch::upsample_bilinear2d(
          input_tensor, {resize_h_, resize_w_}, true);
    }
    if (is_hwc_tensor) {
      im_resize = im_resize.permute({0, 2, 3, 1}).squeeze(0);
    }

    (*input_dict)[TASK_RESULT_KEY] = im_resize;
  }

 private:
  long resize_h_{0};
  long resize_w_{0};
};

OMNI_REGISTER(omniback::Backend, DynamicResizeTensor, "DynamicResizeTensor");
} // namespace torchpipe