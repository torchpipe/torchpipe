
#include <fstream>

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
          throw std::runtime_error("input_tensor should be gpu tensor");
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

void IndexSelectTensor::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& options) {
  parser_v2::ArgsKwargs args_kwargs =
      parser_v2::get_args_kwargs(this, default_cls_name(), params);
  HAMI_ASSERT(args_kwargs.first.size() == 1,
              "Requires exactly 1 argument. Usage: "
              "IndexSelectTensor(weight.pt)/IndexSelectTensor::args=weight.pt");
  std::string device = "cuda";
  str::try_update(args_kwargs.second, "device", device);

  const auto& name = args_kwargs.first.at(0);
  SPDLOG_INFO("load " + name);
  std::ifstream file(name);
  if (!file.good()) {
    throw std::invalid_argument(name + " not exists.");
  }
  file.seekg(0, file.end);
  int length = file.tellg();
  file.seekg(0, file.beg);

  std::vector<char> data(length);
  file.read(data.data(), length);

  device_ = torch::Device(device);

  weight_ = torch::pickle_load(data).toTensor().to(device_);
}

void IndexSelectTensor::forward(const dict& io) {
  auto input = hami::dict_get<torch::Tensor>(io, TASK_DATA_KEY);
  if (device_ != input.device()) {
    input = input.to(device_);
  }
  if (input.sizes().size() == 2 && input.size(0) == 1) {
    input = input.squeeze(0);
  }

  torch::Tensor data_loaded = weight_.index_select(0, input);
  (*io)[TASK_RESULT_KEY] = data_loaded;
}
HAMI_REGISTER_BACKEND(IndexSelectTensor);

void EmbeddingTensor::forward(const dict& io) {
  auto input = hami::dict_get<torch::Tensor>(io, TASK_DATA_KEY);
  if (device_ != input.device()) {
    input = input.to(device_);
  }
  torch::Tensor data_loaded = torch::embedding(
      /*weight=*/weight_,  // 加载的权重矩阵
      /*indices=*/input    // .to(torch::kLong)
  );

  (*io)[TASK_RESULT_KEY] = data_loaded;
}
HAMI_REGISTER_BACKEND(EmbeddingTensor);
}  // namespace torchpipe