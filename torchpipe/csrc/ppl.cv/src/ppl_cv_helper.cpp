#include "torch_utils.hpp"
#include "exception.hpp"
#include "base_logging.hpp"
#include "ppl_cv_helper.hpp"
namespace ipipe {

HWCTensorWrapper::HWCTensorWrapper(dict input_dict, int top, int bottom, int left, int right)
    : input(input_dict) {
  input_tensor = dict_get<torch::Tensor>(input_dict, TASK_DATA_KEY);
  IPIPE_ASSERT(!is_cpu_tensor(input_tensor));
  is_hwc_input = is_hwc(input_tensor);
  input_tensor = img_hwc_guard(input_tensor);
  if (!input_tensor.is_contiguous()) {
    SPDLOG_WARN(
        "the input tensor is not contiguous w.r.t. HWC format; "
        "If you call this backend multiple times with the same input tensor, please in advance "
        "convert the format of the input tensor to contiguous HWC.");
    input_tensor = input_tensor.contiguous();
  }

  int c = input_tensor.sizes()[2];
  IPIPE_ASSERT(c == 3);

  auto options = torch::TensorOptions()
                     .device(torch::kCUDA, -1)
                     .dtype(input_tensor.scalar_type())  // torch::kByte
                     .layout(torch::kStrided)
                     .requires_grad(false);

  output_tensor = torch::empty(
      {top + bottom + input_tensor.sizes()[0], left + right + input_tensor.sizes()[1], c}, options,
      torch::MemoryFormat::Contiguous);

  // assert((img_w + left + right) * c == output_tensor.stride(0));
  assert(input_tensor.sizes()[1] * c == input_tensor.stride(0));
}

HWCTensorWrapper::HWCTensorWrapper(dict input_dict, int target_h, int target_w, bool set_zero)
    : input(input_dict) {
  input_tensor = dict_get<torch::Tensor>(input_dict, TASK_DATA_KEY);
  if (is_cpu_tensor(input_tensor)) {
    input_tensor = input_tensor.cuda();
  }
  is_hwc_input = is_hwc(input_tensor);
  input_tensor = img_hwc_guard(input_tensor);
  if (!input_tensor.is_contiguous()) {
    SPDLOG_WARN(
        "the input tensor is not contiguous w.r.t. HWC format; "
        "If you call this backend multiple times with the same input tensor, please in advance "
        "convert the format of the input tensor to contiguous HWC.");
    input_tensor = input_tensor.contiguous();
  }

  int c = input_tensor.sizes()[2];
  IPIPE_ASSERT(c == 3);

  auto options = torch::TensorOptions()
                     .device(torch::kCUDA, -1)
                     .dtype(input_tensor.scalar_type())  // torch::kByte
                     .layout(torch::kStrided)
                     .requires_grad(false);
  if (set_zero) {
    output_tensor = torch::zeros({target_h, target_w, c}, options);
  } else {
    output_tensor = torch::empty({target_h, target_w, c}, options, torch::MemoryFormat::Contiguous);
  }

  assert(output_tensor.stride(0) == target_w * c);
}

HWCTensorWrapper::HWCTensorWrapper(dict input_dict) : input(input_dict) {
  input_tensor = dict_get<torch::Tensor>(input_dict, TASK_DATA_KEY);
  IPIPE_ASSERT(!is_cpu_tensor(input_tensor));
  is_hwc_input = is_hwc(input_tensor);
  input_tensor = img_hwc_guard(input_tensor);
  if (!input_tensor.is_contiguous()) {
    SPDLOG_WARN(
        "the input tensor is not contiguous w.r.t. HWC format; "
        "If you call this backend multiple times with the same input tensor, please in advance "
        "convert the format of the input tensor to contiguous HWC.");
    input_tensor = input_tensor.contiguous();
  }
  int target_h = input_tensor.sizes()[0];
  int target_w = input_tensor.sizes()[1];
  int c = input_tensor.sizes()[2];
  IPIPE_ASSERT(c == 3);

  auto options = torch::TensorOptions()
                     .device(torch::kCUDA, -1)
                     .dtype(input_tensor.scalar_type())  // torch::kByte
                     .layout(torch::kStrided)
                     .requires_grad(false);

  output_tensor = torch::empty({target_h, target_w, c}, options, torch::MemoryFormat::Contiguous);
  assert(output_tensor.stride(0) == target_w * c);
}

void HWCTensorWrapper::finalize() {
  // try {
  if (!is_hwc_input) {
    output_tensor = output_tensor.permute({2, 0, 1}).unsqueeze(0);
  }
  // } catch (std::exception& e) {
  //   SPDLOG_ERROR("HWCTensorWrapper::~HWCTensorWrapper() failed: {}", e.what());
  //   input = nullptr;
  // }

  if (input) (*input)[TASK_RESULT_KEY] = output_tensor;
}
}  // namespace ipipe