#include "mat_torch/converts.hpp"

#include "helper/mat.hpp"
#include "helper/torch.hpp"
#include <tvm/ffi/container/tensor.h>
#include "omniback/addons/torch/type_traits.h"
#include <torch/torch.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace torchpipe {

inline torch::Tensor cvMat2TorchCPU(const cv::Mat& da) {
  OMNI_ASSERT(
      (da.type() == CV_8UC3 || da.type() == CV_32FC3) && da.isContinuous());

  auto image_tensor = torch::from_blob(
      da.data,
      {da.rows, da.cols, da.channels()},
      da.elemSize1() == 1 ? torch::kByte : torch::kFloat);

  return image_tensor;
}

inline torch::Tensor cvMat2TorchCUDA(const cv::Mat& image) {
  auto re = cvMat2TorchCPU(image);
  return re.to(torch::kCUDA);
}

inline cv::Mat torchTensortoCVMatV2(torch::Tensor tensor, bool deepcopy) { //
  tensor = img_hwc_guard(tensor);
  cv::Mat mat;
  tensor = tensor.to(torch::kCPU).contiguous();

  if (tensor.dtype() == torch::kByte) {
    mat = cv::Mat(
        cv::Size(tensor.size(1), tensor.size(0)),
        CV_8UC(tensor.size(2)),
        tensor.data_ptr<uchar>());
  } else if (tensor.dtype() == torch::kFloat) {
    mat = cv::Mat(
        cv::Size(tensor.size(1), tensor.size(0)),
        CV_32FC(tensor.size(2)),
        tensor.data_ptr<float>());
  } else if (tensor.dtype() == torch::kHalf) {
    tensor = tensor.to(torch::kFloat);
    mat = cv::Mat(
        cv::Size(tensor.size(1), tensor.size(0)),
        CV_32FC(tensor.size(2)),
        tensor.data_ptr<float>());
  } else {
    throw std::runtime_error(
        "unsupported datatype " + std::string(tensor.dtype().name()));
  }

  if (deepcopy) {
    mat = mat.clone();
    OMNI_ASSERT(mat.isContinuous());

    return mat;

  } else
    return mat;
}

void Mat2Tensor::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const omniback::dict& kwargs) {
  omniback::str::try_update(config, "device", device_, {"cpu", "cuda"});
  //   capsule
}

void Mat2Tensor::forward(const omniback::dict& input_dict) {
  auto& input = *input_dict;

  auto iter = input_dict->find(TASK_DATA_KEY);
  OMNI_ASSERT(iter != input_dict->end());
  if (auto opt = iter->second.try_cast<cv::Mat>()) {
    cv::Mat data = opt.value();
    if (device_ == "cpu") {
      if (!data.isContinuous()) {
        SPDLOG_WARN("Mat is not continuous");
        data = data.clone();
      }
      input[TASK_RESULT_KEY] = cvMat2TorchCPU(data).clone();
    } else {
      input[TASK_RESULT_KEY] = cvMat2TorchCUDA(data);
    }

  } else if (auto opt = iter->second.try_cast<std::vector<cv::Mat>>()) {
    std::vector<cv::Mat> data = opt.value();
    std::vector<torch::Tensor> result;
    for (auto d : data) {
      if (device_ == "cpu") {
        if (!d.isContinuous()) {
          SPDLOG_WARN("Mat is not continuous");
          d = d.clone();
        }
        result.emplace_back(cvMat2TorchCPU(d).clone());
      } else
        result.emplace_back(cvMat2TorchCUDA(d));
    }

    input[TASK_RESULT_KEY] = result;
  } else {
    TVM_FFI_THROW(TypeError);
  }
}

OMNI_REGISTER(omniback::Backend, Mat2Tensor);

void Tensor2Mat::forward(const omniback::dict& input_dict) {
  auto& input = *input_dict;

  auto iter = input_dict->find(TASK_DATA_KEY);
  OMNI_ASSERT(iter != input_dict->end());
  auto data = iter->second.cast<torch::Tensor>();
  {
    auto result = torchTensortoCVMatV2(data, true); // true is for 'deepcopy'
    input[TASK_RESULT_KEY] = result;
  }
}
OMNI_REGISTER(omniback::Backend, Tensor2Mat);

} // namespace torchpipe