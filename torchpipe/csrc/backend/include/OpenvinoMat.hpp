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

#pragma once

#include "Backend.hpp"
#include "params.hpp"
#include "prepost.hpp"
struct ModelInstances;
namespace ipipe {

class OpenvinoMat : public Backend {
 public:
  /**
   * @brief
   * @param postprocessor
   * 自定义后处理。自定义网络输出c++批后处理;默认操作是拆分batch维度；
   * 需要实现为 @ref PostProcessor  的子类并注册
   * @param OpenvinoMat::backend 模型转换器。@ref _engine
   * 不存在时，会调用此参数指定的转换器，默认为 @ref Onnx2TensorrtConverter.
   * @param instance_num 实例数目；如果tensorrt
   * engine的profile数目不足以建立足够的实例数，将反序列化多个engine.
   * @param _engine
   类型为`std::shared_ptr<CudaEngineWithRuntime>`, 位于shared_config 中的
   tensorrt 引擎. 如果此tensorrt engine
   已经没有profile供下一个实例使用时，会删除此键值。 如果不存在， 将调用
   @ref OpenvinoMat::backend 指定的转换器产生。
   @param _independent_thread_index 实例序号。 默认为0.
   */
  virtual bool init(const std::unordered_map<std::string, std::string> &,
                    dict shared_config) override;

  /**
   * @brief
   * @param TASK_DATA_KEY
   * 输入数据。网络为单输入输出时，类型为at::Tensor/torch.Tensor,
   * 网络为多输入输出时， 类型为vector/List. 字典序排列.
   * @param[out] TASK_RESULT_KEY 输出类型参照输入类型。 gpu上连续, @ref
   * postprocessor 可修改此类型。网络多输出时按照名称的字典序排列。
   * @ref postprocessor 可自定义输出。需要确保给 TASK_RESULT_KEY 赋值。
   *
   */
  void forward(const std::vector<dict> &) override;

  /**
   * @brief 此实例中模型的最大batch
   *
   */
  virtual uint32_t max() const { return 1; }
  /**
   * @brief 此实例中模型的最大小batch
   *
   */
  virtual uint32_t min() const {
    return 1;
    ;
  };
  ~OpenvinoMat(){

  };

 private:
  std::unique_ptr<Params> params_;

  std::unique_ptr<Backend> backend_;
  std::vector<std::vector<int>> mins_;
  std::vector<std::vector<int>> maxs_;
  std::shared_ptr<ModelInstances> instances_;

  int max_{1};
  int min_{1};

  std::vector<void *> binding_;

  std::vector<bool> change_shape_;

  int profile_index_{0};

  std::vector<uint32_t> new_location_inputs_;
  std::vector<uint32_t> new_location_ouputs_;

  int independent_thread_index_{0};
};
}  // namespace ipipe