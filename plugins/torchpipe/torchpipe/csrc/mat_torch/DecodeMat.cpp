#include "mat_torch/DecodeMat.hpp"

#include "helper/mat.hpp"

#include "opencv2/imgcodecs.hpp"
// #include "opencv2/imgproc.hpp"

namespace torchpipe {

namespace {

cv::Mat cpu_decode(const std::string& data) {
  // Avoid copying data by using cv::Mat with const_cast (imdecode doesn't modify data)
  // cv::Mat signature: Mat(int rows, int cols, int type, void* data, size_t step = AUTO_STEP)
  // For decoding, we create a 1-row Mat with the data
  cv::Mat data_mat(1, static_cast<int>(data.size()), CV_8UC1, 
                   const_cast<void*>(static_cast<const void*>(data.data())));
  return cv::imdecode(data_mat, cv::IMREAD_COLOR);
}
} // namespace
void DecodeMat::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const om::dict& kwargs) {
  //   str::try_update(config, "color", color_);
  //   str::try_update(config, "data_format", data_format_);

  //   OMNI_ASSERT(color_ == "rgb" || color_ == "bgr");
  //   OMNI_ASSERT(data_format_ == "nchw" || data_format_ == "hwc");
}

void DecodeMat::forward(const om::dict& input_dict) {
  auto& input = *input_dict;

  const std::string& data = input.at(TASK_DATA_KEY).cast<std::string>();
 
  auto tensor = cpu_decode(data); // tensor type is Mat
  if (tensor.channels() != 3) {
    SPDLOG_ERROR(
        "only support tensor.channels() == 3. get {}; hxw= {}x{}",
        tensor.channels(),
        tensor.rows,
        tensor.cols);
    return;
  }
  if (tensor.empty()) {
    SPDLOG_ERROR(std::string("DecodeMat: result is empty"));
    return;
  }
  OMNI_ASSERT(tensor.size().width != 0 && tensor.size().height != 0);

  input[TASK_RESULT_KEY] = tensor;
  static const std::string bgr = std::string("bgr");
  input["color"] = bgr;
}

OMNI_REGISTER(om::Backend, DecodeMat);
} // namespace torchpipe