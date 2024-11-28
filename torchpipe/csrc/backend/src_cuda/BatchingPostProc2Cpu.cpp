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

#include <torch/torch.h>

// #include "c10/cuda/CUDAStream.h"
#include "prepost.hpp"
#include "torch_utils.hpp"

#include "Backend.hpp"
#include "dict.hpp"
#include "reflect.h"
#include <torch/torch.h>
namespace ipipe {
/**
 * @brief 数据拷贝到cpu
 * @note 这里发生了显式的流同步
 *
 */
class BatchingPostProc2Cpu : public TorchPostProcessor {
 public:
  void forward(std::vector<torch::Tensor> net_putputs, std::vector<dict> input,
               const std::vector<torch::Tensor>& net_inputs) {
    for (auto& item : net_putputs) {
      item = item.cpu();  // 这种写法也可以， 不过如果多个输出可能同步多次
      // https://github.com/pytorch/pytorch/blob/e10b762537214ad152724d772b72b17a4448f145/aten/src/ATen/native/cuda/Copy.cu#L231
      // 在当前流上
      // item = async2cpu(item); //
    }
    // c10::cuda::getCurrentCUDAStream()
    //     .synchronize(); // 多个输出可以只同步一次； 如果使用.cpu()
    // 则不需要此句

    TorchPostProcessor::forward(net_putputs, input, net_inputs);
  }
};

IPIPE_REGISTER(TorchPostProcessor, BatchingPostProc2Cpu, "cpu");

class BatchingPostProcMax : public TorchPostProcessor {
 public:
  void forward(std::vector<torch::Tensor> net_putputs, std::vector<dict> input,
               const std::vector<torch::Tensor>& net_inputs) {
    IPIPE_ASSERT(net_putputs.size() == 1);
    auto item = net_putputs[0].softmax(1);
    auto result = torch::max(item, 1);
    auto index = std::get<1>(result);
    auto score = std::get<0>(result);

    if (input.size() == 1) {
      std::vector<float> single_result{float(index.item<int64_t>()), score.item<float>()};
      (*input[0])[TASK_RESULT_KEY] = single_result;
    } else {
      index = index.cpu();
      score = score.cpu();
      for (std::size_t i = 0; i < input.size(); ++i) {
        std::vector<float> single_result{float(index[i].item<int64_t>()),
                                         float(score[i].item<float>())};
        (*input[i])[TASK_RESULT_KEY] = single_result;
      }
    }
  }
};

IPIPE_REGISTER(TorchPostProcessor, BatchingPostProcMax, "SoftmaxMax");

class BatchingPostProcSoftmaxCpu : public TorchPostProcessor {
 public:
  void forward(std::vector<torch::Tensor> net_outputs, std::vector<dict> input,
               const std::vector<torch::Tensor>& net_inputs) {
    for (auto& item : net_outputs) {
      if (item.dim() == 2) {
        item = item.softmax(1).cpu();  // 隐式同步
        // item = async2cpu(item);
      }
    }
    // c10::cuda::getCurrentCUDAStream().synchronize(); // 同步cpu数据

    TorchPostProcessor::forward(net_outputs, input, net_inputs);
  }
};

IPIPE_REGISTER(TorchPostProcessor, BatchingPostProcSoftmaxCpu, "softmaxcpu,SoftmaxCpu");

using PostProcessor_at_Tensor = TorchPostProcessor;
IPIPE_REGISTER(TorchPostProcessor, PostProcessor_at_Tensor, "split");

class BatchingPostProcSoftmaxArgMax : public TorchPostProcessor {
 public:
  void forward(std::vector<torch::Tensor> net_outputs, std::vector<dict> input,
               const std::vector<torch::Tensor>& net_inputs) {
    IPIPE_ASSERT(net_outputs.size() == 1);
    torch::Tensor output = net_outputs[0].softmax(1);
    auto max_values_and_indices = torch::max(output, 1);

    torch::Tensor max_values = std::get<0>(max_values_and_indices).cpu();
    torch::Tensor max_indices = std::get<1>(max_values_and_indices).cpu();

    for (std::size_t i = 0; i < input.size(); ++i) {
      float max_score = max_values[i].item<float>();
      int argmax = max_indices[i].item<int>();

      (*input[i])["score"] = max_score;
      (*input[i])["class"] = argmax;
      (*input[i])[TASK_RESULT_KEY] = argmax;
    }
  }
};

IPIPE_REGISTER(TorchPostProcessor, BatchingPostProcSoftmaxArgMax, "SoftmaxArgMax");

}  // namespace ipipe