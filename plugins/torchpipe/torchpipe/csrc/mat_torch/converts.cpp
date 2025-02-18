#include "mat_torch/converts.hpp"

#include "helper/mat_torch.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <torch/torch.h>

namespace torchpipe {

inline torch::Tensor cvMat2TorchCPU(const cv::Mat& image) {
  HAMI_ASSERT(image.type() == CV_8UC3 && (image.isContinuous()));
  return torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, torch::kByte);
}

inline torch::Tensor cvMat2TorchCUDA(const cv::Mat& image) {
  auto re = cvMat2TorchCPU(image);
  return re.to(torch::kCUDA);
}

void Mat2Tensor::init(const std::unordered_map<std::string, std::string>& config,
                      const hami::dict& dict_config) {
  hami::str::try_update(config, "device", device_, {"cpu", "cuda"});
  //   capsule
}

void Mat2Tensor::forward(const hami::dict& input_dict) {
  auto& input = *input_dict;

  auto iter = input_dict->find(TASK_DATA_KEY);
  HAMI_ASSERT(iter != input_dict->end());
  if (iter->second.type() == typeid(cv::Mat)) {
    cv::Mat data = hami::any_cast<cv::Mat>(iter->second);
    if (device_ == "cpu") {
      if (!data.isContinuous()) {
        SPDLOG_WARN("Mat is not continuous");
        data = data.clone();
      }
      input[TASK_RESULT_KEY] = cvMat2TorchCPU(data).clone();
    } else {
      input[TASK_RESULT_KEY] = cvMat2TorchCUDA(data);
    }

  } else if (iter->second.type() == typeid(std::vector<cv::Mat>)) {
    const std::vector<cv::Mat>& data = hami::any_cast<std::vector<cv::Mat>>(iter->second);
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
    SPDLOG_ERROR("unknown type: {}", iter->second.type().name());
    throw std::runtime_error("unknown type: " + std::string(iter->second.type().name()));
  }
}

HAMI_REGISTER(hami::Backend, Mat2Tensor);
}  // namespace torchpipe