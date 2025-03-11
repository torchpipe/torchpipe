#include "mat_torch/CvtColorMat.hpp"

#include "helper/mat.hpp"

#include "helper/task_keys.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

using namespace hami;

namespace torchpipe {

void CvtColorMat::impl_init(
    const std::unordered_map<std::string, std::string>& config_param,
    const dict& dict_config) {
    str::try_update(config_param, "color", color_, VALID_COLOR_SPACE);
}

void CvtColorMat::forward(const dict& input_dict) {
    auto src_color = input_dict->find(TASK_COLOR_KEY);
    HAMI_ASSERT(src_color != input_dict->end(),
                "input dict must contain color");

    std::string input_color = any_cast<std::string>(src_color->second);

    if (input_color == color_) {
        (*input_dict)[TASK_RESULT_KEY] = (*input_dict)[TASK_DATA_KEY];
        return;
    } else {
        HAMI_ASSERT(VALID_COLOR_SPACE.count(input_color) != 0,
                    input_color + " is not supported yet");

        auto input_tensor = hami::dict_get<cv::Mat>(input_dict, TASK_DATA_KEY);
        cv::cvtColor(input_tensor, input_tensor, cv::COLOR_BGR2RGB);

        (*input_dict)[TASK_COLOR_KEY] = color_;
        (*input_dict)[TASK_RESULT_KEY] = input_tensor;
    }
}

HAMI_REGISTER(Backend, CvtColorMat, "CvtColorMat,cvtColorMat");

}  // namespace torchpipe