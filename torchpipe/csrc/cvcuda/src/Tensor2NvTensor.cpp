#include "Tensor2NvTensor.hpp"
#include "reflect.h"
#include <ATen/ATen.h>
// #include <nvcv/Tensor.hpp>
// #include <nvcv/DataType.hpp>

#include "torch_utils.hpp"
#include "cvcuda_helper.hpp"
namespace ipipe {

bool Tensor2NvTensor::init(const std::unordered_map<std::string, std::string> &, dict) {
  return true;
}

void Tensor2NvTensor::forward(dict input_dict) {
  auto &input = *input_dict;
  auto data = dict_get<at::Tensor>(input_dict, TASK_DATA_KEY);
  IPIPE_ASSERT(data.is_cuda());
#if 0
  // CVCUDA only support contiguous tensor at this moment
  std::string data_format;
  auto iter = input.find("data_format");
  if (iter != input.end()) {
    data_format = any_cast<std::string>(iter->second);
  } else if (is_hwc(data)) {
    data_format = "hwc";
  } else if (is_nchw(data)) {
    data_format = "nchw";
  } else {
    throw std::runtime_error("Tensor2NvTensor: Unsupported layout");
  }

  nvcv::Tensor result = toNvcvTensor(data, data_format);
#else
  data = img_hwc_guard(data);
  if (!data.is_contiguous()) {
    data = data.contiguous();
  }
  nvcv::Tensor result = toNvcvTensor(data, "hwc");
#endif
  input[TASK_RESULT_KEY] = result;
}

void NvTensor2Tensor::forward(dict input_dict) {
  auto &input = *input_dict;

  auto data = dict_get<nvcv::Tensor>(input_dict, TASK_DATA_KEY);

  at::Tensor result = fromNvcvTensor(data);
  input[TASK_RESULT_KEY] = result;
}

IPIPE_REGISTER(Backend, Tensor2NvTensor, "Tensor2NvTensor");

IPIPE_REGISTER(Backend, NvTensor2Tensor, "NvTensor2Tensor");

}  // namespace ipipe