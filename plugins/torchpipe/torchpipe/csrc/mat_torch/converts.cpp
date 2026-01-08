#include "mat_torch/converts.hpp"

// #include "helper/torch.hpp"

// #include <torch/torch.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "helper/mat.hpp"

namespace torchpipe{
namespace {
using convert::ImageData ;
cv::Mat ImageData2Mat(const ImageData& img) {
  if (img.data == nullptr) {
    throw std::runtime_error("ImageData::data is null.");
  }

  cv::Mat mat;
  if (img.is_float) {
    // float32
    mat = cv::Mat(
        static_cast<int>(img.rows),
        static_cast<int>(img.cols),
        CV_32FC(static_cast<int>(img.channels)));
    std::memcpy(
        mat.data, img.data, sizeof(float) * img.rows * img.cols * img.channels);
  } else {
    // uint8
    mat = cv::Mat(
        static_cast<int>(img.rows),
        static_cast<int>(img.cols),
        CV_8UC(static_cast<int>(img.channels)));
    std::memcpy(
        mat.data,
        img.data,
        sizeof(uint8_t) * img.rows * img.cols * img.channels);
  }

  return mat;
}

convert::ImageData cvMatToImageData(const cv::Mat& mat) {
  OMNI_ASSERT(mat.type() == CV_8UC3 || mat.type() == CV_32FC3);

  ImageData out;
  out.rows = mat.rows;
  out.cols = mat.cols;
  out.channels = mat.channels();
  out.is_float = (mat.type() == CV_32FC3);

  auto* heap_mat = new cv::Mat(mat);
  if (!heap_mat->isContinuous()) {
    *heap_mat = heap_mat->clone();
  }

  // shallow copy, shares data
  out.data = heap_mat->data;
  out.deleter = [heap_mat](void*) { delete heap_mat; };

  return out;
}
} // 

} // namespace namespace
namespace torchpipe {

using namespace convert;

void Mat2Tensor::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const om::dict& kwargs) {
  om::str::try_update(config, "device", device_, {"cpu", "cuda"});
  //   capsule
}

void Mat2Tensor::forward(const om::dict& input_dict) {
  auto& input = *input_dict;

  auto iter = input_dict->find(TASK_DATA_KEY);
  OMNI_ASSERT(iter != input_dict->end());
  if (auto opt = iter->second.try_cast<cv::Mat>()) {
    cv::Mat data = opt.value();
    if (device_ == "cpu") {
      input[TASK_RESULT_KEY] = imageDataToAnyTorchCPU(cvMatToImageData(data));
    } else {
      input[TASK_RESULT_KEY] = imageDataToAnyTorchGPU(cvMatToImageData(data));
    }

  } else if (auto opt = iter->second.try_cast<std::vector<cv::Mat>>()) {
    std::vector<cv::Mat> data = opt.value();
    std::vector<om::any> result;
    for (auto d : data) {
      if (device_ == "cpu") {
        result.emplace_back(imageDataToAnyTorchCPU(cvMatToImageData(d)));
      } else
        result.emplace_back(imageDataToAnyTorchGPU(cvMatToImageData(d)));
    }

    input[TASK_RESULT_KEY] = result;
  } else {
    TVM_FFI_THROW(TypeError);
  }
}

OMNI_REGISTER(om::Backend, Mat2Tensor);

void Tensor2Mat::forward(const om::dict& input_dict) {
  auto& input = *input_dict;

  auto iter = input_dict->find(TASK_DATA_KEY);
  OMNI_ASSERT(iter != input_dict->end());
  // auto data = iter->second.cast<torch::Tensor>();
  auto data = ImageData2Mat(TorchAny2ImageData(iter->second));
  input[TASK_RESULT_KEY] = data;
}

OMNI_REGISTER(om::Backend, Tensor2Mat);

} // namespace torchpipe