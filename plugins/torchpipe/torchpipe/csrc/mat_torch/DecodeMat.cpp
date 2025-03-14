#include "mat_torch/DecodeMat.hpp"

#include "helper/mat.hpp"

#include "opencv2/imgcodecs.hpp"
// #include "opencv2/imgproc.hpp"

namespace torchpipe {

namespace {

cv::Mat cpu_decode(std::string data) {
    std::vector<char> vectordata(data.begin(), data.end());

    // Check if the data is a JPEG file
    if (vectordata.size() < 2 || vectordata[0] != char(0xFF) ||
        vectordata[1] != char(0xD8)) {
        SPDLOG_ERROR("The data is not a valid JPEG file.");
        return cv::Mat();
    }

    return cv::imdecode(cv::Mat(vectordata), cv::IMREAD_COLOR);
}
}  // namespace
void DecodeMat::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const hami::dict& kwargs) {
    //   str::try_update(config, "color", color_);
    //   str::try_update(config, "data_format", data_format_);

    //   HAMI_ASSERT(color_ == "rgb" || color_ == "bgr");
    //   HAMI_ASSERT(data_format_ == "nchw" || data_format_ == "hwc");
}

void DecodeMat::forward(const hami::dict& input_dict) {
    auto& input = *input_dict;

    if (typeid(std::string) != input.at(TASK_DATA_KEY).type()) {
        throw std::runtime_error(
            std::string("DecodeMat: not support the input type: ") +
            std::string(input[TASK_DATA_KEY].type().name()));
    }
    const std::string* data =
        hami::any_cast<std::string>(&input[TASK_DATA_KEY]);
    HAMI_ASSERT(data && !data->empty());
    auto tensor = cpu_decode(*data);  // tensor type is Mat
    if (tensor.channels() != 3) {
        SPDLOG_ERROR("only support tensor.channels() == 3. get {}",
                     tensor.channels());
        return;
    }
    if (tensor.empty()) {
        SPDLOG_ERROR(std::string("DecodeMat: result is empty"));
        return;
    }
    HAMI_ASSERT(tensor.size().width != 0 && tensor.size().height != 0);

    input[TASK_RESULT_KEY] = tensor;
    static const std::string bgr = std::string("bgr");
    input["color"] = bgr;
}

HAMI_REGISTER(hami::Backend, DecodeMat);
}  // namespace torchpipe