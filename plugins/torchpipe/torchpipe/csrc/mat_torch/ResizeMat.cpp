#include "mat_torch/ResizeMat.hpp"

#include "helper/mat.hpp"
#include "omniback/helper/string.hpp"
#include "opencv2/imgproc.hpp"
namespace torchpipe {

void ResizeMat::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const omniback::dict& kwargs) {
  resize_h_ = omniback::str::str2int<size_t>(config, "resize_h");
  resize_w_ = omniback::str::str2int<size_t>(config, "resize_w");
}

void ResizeMat::forward(const omniback::dict& input_dict) {
  auto& input = *input_dict;

  auto data = input[TASK_DATA_KEY].cast<cv::Mat>();

  cv::Mat im_resize;
  cv::resize(data, im_resize, cv::Size(resize_w_, resize_h_));

  if (im_resize.cols == 0 || im_resize.rows == 0 || im_resize.channels() == 0) {
    SPDLOG_ERROR(
        "im_resize.cols={}, im_resize.rows={}, im_resize.channels={}",
        im_resize.cols,
        im_resize.rows,
        im_resize.channels());
    return;
  }

  input[TASK_RESULT_KEY] = im_resize;
}

OMNI_REGISTER(omniback::Backend, ResizeMat);

void LetterBoxMat::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const omniback::dict& kwargs) {
  target_h_ = omniback::str::str2int<size_t>(config, "resize_h");
  target_w_ = omniback::str::str2int<size_t>(config, "resize_w");

  std::string pad_val_str =
      config.count("pad_val") ? config.at("pad_val") : "0,0,0";

  if (pad_val_str.find(',') != std::string::npos) {
    std::vector<std::string> vals = omniback::str::str_split(pad_val_str, ',');
    if (vals.size() >= 3) {
      pad_val_ = cv::Scalar(
          std::stoi(vals[0]), std::stoi(vals[1]), std::stoi(vals[2]));
    }
  } else {
    int val = std::stoi(pad_val_str);
    pad_val_ = cv::Scalar(val, val, val);
  }
}

void LetterBoxMat::forward(const omniback::dict& input_dict) {
  auto& input = *input_dict;

  cv::Mat src = input[TASK_DATA_KEY].cast<cv::Mat>();
  if (src.empty()) {
    SPDLOG_ERROR("LetterBoxMat: input image is empty");
    return;
  }

  // 计算缩放比例 (取宽高比例的最小值)
  double scale = std::min(
      static_cast<double>(target_w_) / src.cols,
      static_cast<double>(target_h_) / src.rows);

  // 计算缩放后的尺寸
  int new_w = std::round(src.cols * scale);
  int new_h = std::round(src.rows * scale);

  // 计算居中偏移量
  int offset_x = (target_w_ - new_w) / 2;
  int offset_y = (target_h_ - new_h) / 2;

  // 创建目标图像
  cv::Mat dst(target_h_, target_w_, src.type(), pad_val_);

  // 缩放原始图像
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_w, new_h));

  // 将缩放后的图像复制到中心位置
  cv::Rect roi(offset_x, offset_y, new_w, new_h);
  resized.copyTo(dst(roi));

  // 输出结果
  input[TASK_RESULT_KEY] = dst;

  // 输出缩放比例和偏移量
  input["scale"] = float(scale);
  input["offset"] = std::make_pair(offset_x, offset_y);
}

OMNI_REGISTER(omniback::Backend, LetterBoxMat);

void TopLeftResizeMat::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const omniback::dict& kwargs) {
  target_h_ = omniback::str::str2int<size_t>(config, "resize_h");
  target_w_ = omniback::str::str2int<size_t>(config, "resize_w");

  std::string pad_val_str =
      config.count("pad_val") ? config.at("pad_val") : "0,0,0";

  if (pad_val_str.find(',') != std::string::npos) {
    std::vector<std::string> vals = omniback::str::str_split(pad_val_str, ',');
    if (vals.size() >= 3) {
      pad_val_ = cv::Scalar(
          std::stoi(vals[0]), std::stoi(vals[1]), std::stoi(vals[2]));
    }
  } else {
    int val = std::stoi(pad_val_str);
    pad_val_ = cv::Scalar(val, val, val);
  }
}

void TopLeftResizeMat::forward(const omniback::dict& input_dict) {
  auto& input = *input_dict;

  cv::Mat src = input[TASK_DATA_KEY].cast<cv::Mat>();
  if (src.empty()) {
    SPDLOG_ERROR("TopLeftResizeMat: input image is empty");
    return;
  }

  // 计算缩放比例 (取宽高比例的最小值)
  double scale = std::min(
      static_cast<double>(target_w_) / src.cols,
      static_cast<double>(target_h_) / src.rows);

  // 计算缩放后的尺寸
  int new_w = std::round(src.cols * scale);
  int new_h = std::round(src.rows * scale);

  // 左上角无偏移
  int offset_x = 0;
  int offset_y = 0;

  // 创建目标图像
  cv::Mat dst(target_h_, target_w_, src.type(), pad_val_);

  // 缩放原始图像
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_w, new_h));

  // 将缩放后的图像复制到左上角
  cv::Rect roi(offset_x, offset_y, new_w, new_h);
  resized.copyTo(dst(roi));

  // 输出结果
  input[TASK_RESULT_KEY] = dst;

  // 输出缩放比例和偏移量
  input["scale"] = float(scale);
  input["offset"] = std::make_pair(offset_x, offset_y);
}

OMNI_REGISTER(omniback::Backend, TopLeftResizeMat);

// def postprocess(results, meta):
//     scale = meta['scale']
//     offset_x, offset_y = meta['offset']

// #还原原始坐标
//     for box in results.boxes:
// #调整坐标到缩放后的图像位置
//         box.x1 = (box.x1 - offset_x) / scale
//         box.y1 = (box.y1 - offset_y) / scale
//         box.x2 = (box.x2 - offset_x) / scale
//         box.y2 = (box.y2 - offset_y) / scale
} // namespace torchpipe