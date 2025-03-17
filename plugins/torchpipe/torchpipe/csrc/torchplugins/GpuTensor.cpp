
#include "torchplugins/GpuTensor.hpp"
#include "helper/task_keys.hpp"
#include "helper/torch.hpp"

using namespace hami;

namespace torchpipe {

/**
 * @brief cpu->gpu
 */

class GpuTensor : public hami::BackendOne {
   public:
    void forward(const dict& input_output) override {
        auto data = dict_gets<torch::Tensor>(input_output, TASK_DATA_KEY);
        for (auto& item : data) {
            if (item.is_cpu()) {
                item = item.cuda();
            }
        }
        if (data.size() == 1)
            (*input_output)[TASK_RESULT_KEY] = data[0];
        else
            (*input_output)[TASK_RESULT_KEY] = data;
    }
};
HAMI_REGISTER(Backend, GpuTensor);

/**
 * @brief gpu->cpu
 */
class CpuTensor : public BackendOne {
   public:
    /**
     * @brief cpu->gpu
     * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] =
     * input[TASK_DATA_KEY].cuda()
     */
    virtual void forward(const dict& input_dict) override {
        auto& input = *input_dict;

        if (input[TASK_DATA_KEY].type() == typeid(torch::Tensor)) {
            torch::Tensor input_tensor =
                any_cast<torch::Tensor>(input[TASK_DATA_KEY]);
            if (!input_tensor.is_cuda()) {
                SPDLOG_ERROR("input_tensor should be gpu tensor");
                throw std::runtime_error("input_tensor should be gpu tensor");
            }

            input[TASK_RESULT_KEY] = input_tensor.cpu();
        } else if (input[TASK_DATA_KEY].type() ==
                   typeid(std::vector<torch::Tensor>)) {
            std::vector<torch::Tensor> input_tensor =
                any_cast<std::vector<torch::Tensor>>(input[TASK_DATA_KEY]);
            for (auto& item : input_tensor) {
                if (item.is_cuda()) {
                    item = item.cpu();
                } else {
                    SPDLOG_ERROR("input_tensor should be gpu tensor");
                    throw std::runtime_error(
                        "input_tensor should be gpu tensor");
                }
            }
            input[TASK_RESULT_KEY] = input_tensor;
        } else {
            SPDLOG_ERROR(
                "CpuTensor: torch::Tensor/std::vector<torch::Tensor> needed; "
                "error input type: " +
                std::string(input[TASK_DATA_KEY].type().name()));
            throw std::runtime_error(
                "CpuTensor: torch::Tensor/std::vector<torch::Tensor> needed; "
                "error input type: " +
                std::string(input[TASK_DATA_KEY].type().name()));
        }
    }
};

HAMI_REGISTER(hami::Backend, CpuTensor, "CpuTensor");
}  // namespace torchpipe