#include "mat_torch/ResizeMat.hpp"

#include "helper/mat.hpp"
#include "opencv2/imgproc.hpp"

namespace torchpipe {

void ResizeMat::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const hami::dict& dict_config) {
    resize_h_ = hami::str::str2int<size_t>(config, "resize_h");
    resize_w_ = hami::str::str2int<size_t>(config, "resize_w");
}

void ResizeMat::forward(const hami::dict& input_dict) {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(cv::Mat)) {
        SPDLOG_ERROR("ResizeMat: error input type: " +
                     std::string(input[TASK_DATA_KEY].type().name()));
        return;
    }
    auto data = hami::any_cast<cv::Mat>(input[TASK_DATA_KEY]);

    cv::Mat im_resize;
    cv::resize(data, im_resize, cv::Size(resize_w_, resize_h_));

    if (im_resize.cols == 0 || im_resize.rows == 0 ||
        im_resize.channels() == 0) {
        SPDLOG_ERROR(
            "im_resize.cols={}, im_resize.rows={}, im_resize.channels={}",
            im_resize.cols, im_resize.rows, im_resize.channels());
        return;
    }

    input[TASK_RESULT_KEY] = im_resize;
}

HAMI_REGISTER(hami::Backend, ResizeMat);

}  // namespace torchpipe