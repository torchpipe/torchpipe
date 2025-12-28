#include "mat_torch/DecodeMat.hpp"

#include "helper/mat.hpp"

#include "opencv2/imgcodecs.hpp"
// #include "opencv2/imgproc.hpp"

namespace torchpipe {

namespace {

cv::Mat cpu_decode(std::string data) {
  std::vector<char> vectordata(data.begin(), data.end());

  // Check if the data is a JPEG file
  // if (vectordata.size() < 2 || vectordata[0] != char(0xFF) ||
  //     vectordata[1] != char(0xD8)) {
  //     SPDLOG_ERROR("The data is not a valid JPEG file.");
  //     return cv::Mat();
  // }

  return cv::imdecode(cv::Mat(vectordata), cv::IMREAD_COLOR);
}
} // namespace
void DecodeMat::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const omniback::dict& kwargs) {
  //   str::try_update(config, "color", color_);
  //   str::try_update(config, "data_format", data_format_);

  //   OMNI_ASSERT(color_ == "rgb" || color_ == "bgr");
  //   OMNI_ASSERT(data_format_ == "nchw" || data_format_ == "hwc");
}

void DecodeMat::forward(const omniback::dict& input_dict) {
  auto& input = *input_dict;

  std::string data = input.at(TASK_DATA_KEY).cast<std::string>();
 
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

OMNI_REGISTER(omniback::Backend, DecodeMat);
} // namespace torchpipe