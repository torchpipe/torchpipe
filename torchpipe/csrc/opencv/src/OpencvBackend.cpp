// Copyright 2021-2024 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "OpencvBackend.hpp"

#include <memory>
#include <vector>
#include "base_logging.hpp"
#include "params.hpp"
#include "exception.hpp"

#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "ipipe_utils.hpp"
#include "base_logging.hpp"
// #include <c10/util/Type.h>

#define PALIGN_UP(x, align) ((x + (align - 1)) & ~(align - 1))

#include "reflect.h"
namespace ipipe {

static inline const std::string thread_id_string() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  return ss.str();
}
cv::Mat load_mat(std::string file_name, int h, int w) {
  // std::ifstream in_file(file_name.c_str());
  std::ifstream ifs(file_name, std::ios::binary);
  std::vector<char> data_buffer;
  data_buffer.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

  cv::Mat mat(h, w, CV_8UC3, data_buffer.data());
  return mat;
}
cv::Mat ResizeImgLow(const cv::Mat& im, int min_len, int max_len, int align) {
  int h = im.rows;
  int w = im.cols;

  int w_new = min_len, h_new = max_len;
  if (h > w) {
    h_new = int(float(w_new) / w * h);
    h_new = std::min(PALIGN_UP(h_new, align), max_len);
  } else if (h < w) {
    w_new = int(float(h_new) / h * w);
    w_new = std::min(PALIGN_UP(w_new, align), max_len);
  }
  if (h == h_new && w == w_new) {
    return im;
  }
  cv::Mat im_resize;
  cv::resize(im, im_resize, cv::Size(w_new, h_new));
  return im_resize;
}

// cv::Mat cpu_decode(std::vector<char> vectordata) {
//   auto da = cv::imdecode(cv::Mat(vectordata), cv::IMREAD_COLOR);
//   return da;
// }

cv::Mat cpu_decode(std::string data) {
  std::vector<char> vectordata(data.begin(), data.end());
  return cv::imdecode(cv::Mat(vectordata), cv::IMREAD_COLOR);
}

void DecodeMat::forward(dict input_dict) {
  auto& input = *input_dict;

  if (typeid(std::string) != input[TASK_DATA_KEY].type()) {
    throw std::runtime_error(std::string("DecodeMat: not support the input type: ") +
                             ipipe::local_demangle(input[TASK_DATA_KEY].type().name()));
  }
  const std::string* data = any_cast<std::string>(&input[TASK_DATA_KEY]);
  IPIPE_ASSERT(data && !data->empty());
  auto tensor = cpu_decode(*data);  // tensor type is Mat
  if (tensor.channels() != 3) {
    SPDLOG_ERROR("only support tensor.channels() == 3. get {}", tensor.channels());
    return;
  }
  if (tensor.empty()) {
    SPDLOG_ERROR(std::string("DecodeMat: result is empty"));
    return;
  }
  IPIPE_ASSERT(tensor.size().width != 0 && tensor.size().height != 0);

  input[TASK_RESULT_KEY] = tensor;
  static const std::string bgr = std::string("bgr");
  input["color"] = bgr;
}

IPIPE_REGISTER(Backend, DecodeMat, "DecodeMat");

#ifndef DOXYGEN_SHOULD_SKIP_THIS
bool FixRatioResizeMat::init(const std::unordered_map<std::string, std::string>& config_param,
                             dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({}, {"max_w", "max_h"}, {}, {}));
  if (!params_->init(config_param)) return false;

  TRACE_EXCEPTION(max_w_ = std::stoi(params_->at("max_w")));
  TRACE_EXCEPTION(max_h_ = std::stoi(params_->at("max_h")));

  return true;
}

void FixRatioResizeMat::forward(const std::vector<dict>& input_dicts) {
  for (auto input_dict : input_dicts) {
    params_->check_and_update(input_dict);
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
      SPDLOG_ERROR("FixRatioResizeMat: error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      continue;
    }
    auto data = any_cast<cv::Mat>(input[TASK_DATA_KEY]);

    float ratio = float(data.cols) / float(data.rows);
    int resize_h;
    int resize_w;
    if (max_h_ * ratio <= max_w_) {
      resize_w = std::min(int(ceil(max_h_ * ratio)), max_w_);
      resize_h = max_h_;
    } else {
      resize_w = max_w_;
      resize_h = std::min(int(ceil(max_w_ / ratio)), max_h_);
    }
    cv::Mat resize_img;
    float x_ratio = data.cols * 1.0f / resize_w;
    float y_ratio = data.rows * 1.0f / resize_h;
    std::function<std::pair<float, float>(float, float)> inverse_trans = [x_ratio, y_ratio](
                                                                             float x, float y) {
      return std::pair<float, float>(x_ratio * x, y_ratio * y);
    };
    input["inverse_trans"] = inverse_trans;
    if (resize_w == data.cols && resize_h == data.rows) {
      resize_img = data;
    } else
      cv::resize(data, resize_img, cv::Size(resize_w, resize_h), 0.f, 0.f, cv::INTER_LINEAR);

    input[TASK_RESULT_KEY] = resize_img;
  }
}

IPIPE_REGISTER(Backend, FixRatioResizeMat, "FixRatioResizeMat");
#endif

bool BatchFixHLimitW::init(const std::unordered_map<std::string, std::string>& config_param,
                           dict dict_config) {
  params_ = std::unique_ptr<Params>(
      new Params({{"align", "32"}}, {"resize_h", "BatchFixHLimitW::max", "max_w"}, {}, {}));
  if (!params_->init(config_param)) return false;
  TRACE_EXCEPTION(max_ = std::stoi(params_->at("BatchFixHLimitW::max")));
  TRACE_EXCEPTION(max_w_ = std::stoi(params_->at("max_w")));
  TRACE_EXCEPTION(resize_h_ = std::stoi(params_->at("resize_h")));
  TRACE_EXCEPTION(align_ = std::stoi(params_->at("align")));
  if (max_w_ % align_ != 0) {
    SPDLOG_ERROR("max_w_%align_ != 0");
    return false;
  }

  if (max_ >= 1 && max_ < UINT32_MAX) {
    return true;
  } else {
    SPDLOG_ERROR("not satisfied: max_ > 1 && max_ < UINT32_MAX: max = {}", max_);
    return false;
  }
}

void BatchFixHLimitW::forward(const std::vector<dict>& input_dicts) {
  int target_w = 0;
  std::string col_row;
  for (auto input_dict : input_dicts) {
    // params_->check_and_update(input_dict);
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
      SPDLOG_ERROR("BatchFixHLimitW: error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("BatchFixHLimitW: error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    cv::Mat data = any_cast<cv::Mat>(input[TASK_DATA_KEY]);
    col_row += "[" + std::to_string(data.cols) + "," + std::to_string(data.rows) + "," +
               std::to_string(data.cols * 1.0 / data.rows) + "]";
    target_w = std::max(target_w, int(ceilf(resize_h_ * data.cols) / float(data.rows)));
    if (target_w % align_ != 0) target_w = (target_w / 32 + 1) * 32;
    if (max_w_ > 0 && target_w > max_w_) {
      target_w = max_w_;
    }
  }
  // SPDLOG_DEBUG(col_row);
  for (auto input_dict : input_dicts) {
    auto& input = *input_dict;
    input[TASK_RESULT_KEY] = input[TASK_DATA_KEY];
    input["max_w"] = std::to_string(target_w);
    input["resize_h"] = std::to_string(resize_h_);
  }
}
IPIPE_REGISTER(Backend, BatchFixHLimitW, "BatchFixHLimitW");

bool cvtColorMat::init(const std::unordered_map<std::string, std::string>& config_param,
                       dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({}, {"color"}, {}, {}));
  if (!params_->init(config_param)) return false;

  TRACE_EXCEPTION(color_ = params_->at("color"));

  if ((color_ != "rgb") && color_ != "bgr") {
    SPDLOG_ERROR("error: init: color: " + color_);
    return false;
  }

  return true;
}

void cvtColorMat::forward(const std::vector<dict>& input_dicts) {
  for (auto input_dict : input_dicts) {
    std::string color_data;
    TRACE_EXCEPTION(color_data = any_cast<std::string>(input_dict->at("color")));
    auto& input = *input_dict;
    if (color_data == color_) {
      input[TASK_RESULT_KEY] = input[TASK_DATA_KEY];
      continue;
    }

    auto data = any_cast<cv::Mat>(input[TASK_DATA_KEY]);
    IPIPE_ASSERT(data.channels() == 3);
    cv::Mat output;
    cv::cvtColor(data, output, cv::COLOR_BGR2RGB, 3);
    input[TASK_RESULT_KEY] = output;
    input["color"] = color_;
  }
}

IPIPE_REGISTER(Backend, cvtColorMat, "cvtColorMat,CvtColorMat");

class PerspectiveTransformMats : public SingleBackend {
 public:
  void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
      SPDLOG_ERROR("need cv::Mat, get: " +
                   ipipe::local_demangle(input[TASK_DATA_KEY].type().name()));
      return;
    }
    auto iter = input_dict->find(TASK_BOX_KEY);
    if (iter == input_dict->end()) {
      SPDLOG_ERROR("PerspectiveTransformMats: TASK_BOX_KEY not found.");
      return;
    }

    std::vector<std::vector<std::vector<int>>> boxes =
        any_cast<std::vector<std::vector<std::vector<int>>>>(iter->second);
    cv::Mat img = any_cast<cv::Mat>(input[TASK_DATA_KEY]);
    std::vector<cv::Mat> cropped_imgs;

    for (const auto& points : boxes) {
      int img_crop_width =
          int(sqrt(pow(points[0][0] - points[1][0], 2) + pow(points[0][1] - points[1][1], 2)));
      int img_crop_height =
          int(sqrt(pow(points[0][0] - points[3][0], 2) + pow(points[0][1] - points[3][1], 2)));

      cv::Point2f pts_std[4];
      pts_std[0] = cv::Point2f(0., 0.);
      pts_std[1] = cv::Point2f(img_crop_width, 0.);
      pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
      pts_std[3] = cv::Point2f(0.f, img_crop_height);

      cv::Point2f pointsf[4];
      pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
      pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
      pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
      pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

      cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

      cv::Mat dst_img;
      cv::warpPerspective(img, dst_img, M, cv::Size(img_crop_width, img_crop_height),
                          cv::BORDER_REPLICATE);
      cropped_imgs.push_back(dst_img);
    }
    input[TASK_RESULT_KEY] = cropped_imgs;
  }
};

IPIPE_REGISTER(Backend, PerspectiveTransformMats, "PerspectiveTransformMats");

#ifndef DOXYGEN_SHOULD_SKIP_THIS
class CropResizeMat : public SingleBackend {
 private:
  int resize_h_;
  int resize_w_;
  std::unique_ptr<Params> params_;
  float crop_ratio_ = 0.0;

 public:
  bool init(const std::unordered_map<std::string, std::string>& config_param,
            dict dict_config) override {
    params_ = std::unique_ptr<Params>(
        new Params({{"enlarge_ratio", "0"}}, {"resize_h", "resize_w"}, {}, {}));
    if (!params_->init(config_param)) return false;

    TRACE_EXCEPTION(resize_h_ = std::stoi(params_->at("resize_h")));
    TRACE_EXCEPTION(resize_w_ = std::stoi(params_->at("resize_w")));
    TRACE_EXCEPTION(crop_ratio_ = std::stof(params_->at("enlarge_ratio")));
    if (resize_h_ > 1024 * 1024 || resize_w_ > 1024 * 1024 || resize_h_ < 1 || resize_w_ < 1) {
      SPDLOG_ERROR("CropResizeMat: illigle h or w: h=" + std::to_string(resize_h_) +
                   "w=" + std::to_string(resize_w_));
      return false;
    }

    return true;
  }
  void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
      SPDLOG_ERROR("CropResizeMat: error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      return;
    }
    auto iter = input_dict->find(TASK_BOX_KEY);
    if (iter == input_dict->end()) {
      SPDLOG_ERROR("CropResizeMat: TASK_BOX_KEY not found.");
      return;
    }

    cv::Mat img = any_cast<cv::Mat>(input[TASK_DATA_KEY]);

    std::vector<float> box = any_cast<std::vector<float>>(iter->second);
    assert(!box.empty());

    const float* data = box.data();
    float x1 = (data[0]);
    float y1 = (data[1]);
    float x2 = (data[2]);
    float y2 = (data[3]);

    // if (x1 <= 1 && y1 <= 1 && x2 <= 1 && y2 <= 1)
    {
      x1 = int(x1 * img.cols);
      y1 = int(y1 * img.rows);
      x2 = int(x2 * img.cols);
      y2 = int(y2 * img.rows);
    }
    const float w = x2 - x1 + 1;
    const float h = y2 - y1 + 1;
    const int crop_w = crop_ratio_ * w;  // 此处为了和原版结果保持一致
    const int crop_h = crop_ratio_ * h;

    x1 = std::max(0.f, (x1 - crop_w));  // 此处为了和原版结果保持一致
    y1 = std::max(0.f, (y1 - crop_h));
    x2 = std::min(img.cols, int(x2 + crop_w));
    y2 = std::min(img.rows, int(y2 + crop_h));
    if (x1 >= x2 || y1 >= y2) {
      SPDLOG_ERROR("CropResizeMat: x1 >= x2 || y1 >= y2: {} {} {} {} data = {} {} {} {}", x1, x2,
                   y1, y2, data[0], data[1], data[2], data[3]);
      return;
    }

    cv::Mat img_cropped = img(cv::Range(int(y1), int(y2)), cv::Range(int(x1), int(x2)));
    cv::resize(img_cropped, img_cropped, cv::Size(resize_w_, resize_h_));

    // cropped_tensors.push_back(cropped);

    // input["cropped"] = cropped_tensors;
    input[TASK_RESULT_KEY] = img_cropped;

    // input[TASK_RESULT_KEY] = cropped_imgs;
  }
};

IPIPE_REGISTER(Backend, CropResizeMat, "CropResizeMat");
#endif

/**
 * @brief 保存cv::Mat 到指定目录。用于调试，可能严重影响性能。 参考
 *  @ref SaveTensor.
 *
 */
class SaveMat : public SingleBackend {
 private:
  std::unique_ptr<Params> params_;

 public:
  /**
   * @brief
   *
   * @param save_dir 图片保存路径。需要预先存在。
   */
  bool init(const std::unordered_map<std::string, std::string>& config_param,
            dict dict_config) override {
    params_ = std::unique_ptr<Params>(new Params({{"node_name", ""}}, {"save_dir"}, {}, {}));
    if (!params_->init(config_param)) return false;

    std::ifstream file(params_->at("save_dir").c_str());
    if (!file.good()) {
      SPDLOG_ERROR("SaveMat: dir " + params_->at("save_dir") + " not exists.");
      throw std::invalid_argument("dir " + params_->at("save_dir") + " not exists.");
    }
    return true;
  }

  /**
   * @brief 命名由线程id，thread_local 的 index 决定，故进程内维一。
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY]
   */
  void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
      SPDLOG_ERROR("SaveMat: cv::Mat needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("SaveMat: cv::Mat needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<cv::Mat>(input[TASK_DATA_KEY]);

    const std::string& save_dir = params_->at("save_dir");
    if (!save_dir.empty()) {
      thread_local int index_ = 0;

      thread_local const auto base_save_name =
          save_dir + "/" + params_->get("node_name", "") + "_" + thread_id_string() + "_";

      auto save_name = base_save_name + std::to_string(index_) + ".png";
      cv::imwrite(save_name, input_tensor);
      // torch::save(
      //     input_tensor, base_save_name + std::to_string(index_) + ".pt");
      index_++;
      SPDLOG_WARN("image saved for debug: " + save_name +
                  " . Note that dumping affect the performance.");
    }

    input[TASK_RESULT_KEY] = input[TASK_DATA_KEY];
  }
};

IPIPE_REGISTER(Backend, SaveMat, "SaveMat");

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <typename BaseType, typename InClass, typename OutClass>
class VectorAdapter : public SingleBackend {
  std::unique_ptr<Backend> backend_;
  bool init(const std::unordered_map<std::string, std::string>& config_param,
            dict dict_config) override {
    backend_ = std::make_unique<BaseType>();
    auto init_result = backend_->init(config_param, dict_config);
    if (init_result && backend_->max() == 1) return true;
    return false;
  }
  void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(std::vector<InClass>)) {
      SPDLOG_ERROR("MultipleAdapter: error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("SaveMat: error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    std::vector<InClass> input_tensor = any_cast<std::vector<InClass>>(input[TASK_DATA_KEY]);
    std::vector<OutClass> out_tensor;
    for (std::size_t i = 0; i < input_tensor.size(); ++i) {
      input[TASK_DATA_KEY] = input_tensor[i];
      backend_->forward({input_dict});
      OutClass result = any_cast<OutClass>(input[TASK_RESULT_KEY]);
      input.erase(TASK_RESULT_KEY);
      out_tensor.push_back(result);
    }
    input[TASK_DATA_KEY] = input_tensor;
    input[TASK_RESULT_KEY] = out_tensor;
  }
};
using SaveMats = VectorAdapter<SaveMat, cv::Mat, cv::Mat>;
IPIPE_REGISTER(Backend, SaveMats, "SaveMats");
#endif
}  // namespace ipipe