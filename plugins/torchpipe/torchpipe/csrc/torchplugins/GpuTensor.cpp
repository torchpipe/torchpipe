
#include "torchplugins/GpuTensor.hpp"
#include "helper/task_keys.hpp"
#include "helper/torch.hpp"

using namespace hami;

namespace torchpipe {

/**
 * @brief cpu->gpu
 */
class GpuTensor : public SingleBackend {
 public:
  /**
   * @brief cpu->gpu
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY].cuda()
   */
  virtual void forward(const dict& input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(torch::Tensor)) {
      SPDLOG_ERROR("GpuTensor: torch::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("GpuTensor: torch::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<torch::Tensor>(input[TASK_DATA_KEY]);

    if (input_tensor.is_cpu()) {
      // SPDLOG_ERROR("input_tensor should be cpu tensor");
      // throw std::runtime_error("input_tensor should be cpu tensor");
      input[TASK_RESULT_KEY] = input_tensor;
    }

    input[TASK_RESULT_KEY] = input_tensor.cuda();
  }

 private:
};

HAMI_REGISTER(hami::Backend, GpuTensor, "GpuTensor");

/**
 * @brief gpu->cpu
 */
class CpuTensor : public SingleBackend {
 public:
  /**
   * @brief cpu->gpu
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY].cuda()
   */
  virtual void forward(const dict& input_dict) override {
    auto& input = *input_dict;

    if (input[TASK_DATA_KEY].type() == typeid(torch::Tensor)) {
      torch::Tensor input_tensor = any_cast<torch::Tensor>(input[TASK_DATA_KEY]);
      if (!input_tensor.is_cuda()) {
        SPDLOG_ERROR("input_tensor should be gpu tensor");
        throw std::runtime_error("input_tensor should be gpu tensor");
      }

      input[TASK_RESULT_KEY] = input_tensor.cpu();
    } else if (input[TASK_DATA_KEY].type() == typeid(std::vector<torch::Tensor>)) {
      std::vector<torch::Tensor> input_tensor =
          any_cast<std::vector<torch::Tensor>>(input[TASK_DATA_KEY]);
      for (auto& item : input_tensor) {
        if (item.is_cuda()) {
          item = item.cpu();
        } else {
          SPDLOG_ERROR("input_tensor should be gpu tensor");
          throw std::runtime_error("input_tensor should be gpu tensor");
        }
      }
      input[TASK_RESULT_KEY] = input_tensor;
    } else {
      SPDLOG_ERROR(
          "CpuTensor: torch::Tensor/std::vector<torch::Tensor> needed; error input type: " +
          std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error(
          "CpuTensor: torch::Tensor/std::vector<torch::Tensor> needed; error input type: " +
          std::string(input[TASK_DATA_KEY].type().name()));
    }
  }
};

HAMI_REGISTER(hami::Backend, CpuTensor, "CpuTensor");
}  // namespace torchpipe