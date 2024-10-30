#include "prepost.hpp"
#ifndef NDEBUG
#include "base_logging.hpp"
#endif

namespace ipipe {

template <>
void PostProcessor<torch::Tensor>::forward(std::vector<torch::Tensor> net_outputs,
                                           std::vector<dict> inputs,
                                           const std::vector<torch::Tensor>& net_inputs) {
  SPDLOG_DEBUG("PostProcessor input size:{}, request size:{}, output[0] size[0]:{}", inputs.size(),
               get_request_size(inputs), net_outputs[0].sizes()[0]);
  if (inputs.size() == 1) {
    if (net_outputs.size() == 1)
      (*inputs[0])[TASK_RESULT_KEY] = net_outputs[0];
    else
      (*inputs[0])[TASK_RESULT_KEY] = net_outputs;
    return;
  } else if (net_outputs[0].sizes()[0] > inputs.size()) {
    std::vector<uint32_t> shapes{0};
    for (const auto& item : inputs) {
      shapes.push_back(get_request_size(item));
    }
    IPIPE_ASSERT(std::accumulate(shapes.begin(), shapes.end(), 0) == net_outputs[0].sizes()[0]);
    // 累加
    std::partial_sum(shapes.begin(), shapes.end(), shapes.begin());

    for (std::size_t i = 0; i < inputs.size(); ++i) {
      std::vector<torch::Tensor> single_result;
      for (const auto& item : net_outputs) {
        single_result.push_back(item.index({torch::indexing::Slice(shapes[i], shapes[i + 1])}));
      }
      if (single_result.size() == 1) {
        (*inputs[i])[TASK_RESULT_KEY] = single_result[0];  // 返回torch::Tensor
      } else
        (*inputs[i])[TASK_RESULT_KEY] = single_result;  // 返回std::vector<torch::Tensor>
    }
  } else {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      std::vector<torch::Tensor> single_result;

      for (const auto& item : net_outputs) {
        // IPIPE_ASSERT(item.sizes()[0] == inputs.size());
        single_result.push_back(item[i].unsqueeze(0));
      }
      if (single_result.size() == 1) {
        (*inputs[i])[TASK_RESULT_KEY] = single_result[0];
      } else
        (*inputs[i])[TASK_RESULT_KEY] = single_result;
    }
  }
}
};  // namespace ipipe