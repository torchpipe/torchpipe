// Copyright 2021-2023 NetEase.
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
#include "base_logging.hpp"
#include <memory>
#include <string>
#include <vector>
#include "Backend.hpp"
#include "dict.hpp"
#include "event.hpp"
#include "MultipleInstances.hpp"
#include "params.hpp"
#include "reflect.h"
#include "threadsafe_list.hpp"
#include "time_utils.hpp"
#include "RangeMerger.hpp"
#include "RuningState.hpp"
#include "exception.hpp"

namespace ipipe {

/**
 * @brief (WIP)实现单节点调度，凑batch的功能。配合默认子后端 RangeMerger
 * 能实现分组的功能。配合 RangeMerger 的默认子后端 MultipleInstances
 实现多实例功能。
 * 实际默认调用后端链：``ScoreSchedule[RangeMerger[MultipleInstances[InstanceHandler[backend]]]]``.
 *
 * **使用示例:**
  ```
  # 这里仅以单节点toml配置文件方式展示使用，其他方式使用同理：
  [resnet]
  max="1"
  min="4"
  model = "batch-1.onnx" # or resnet18_merge
  instance_num = 2
  next = "postprocess"
  ```
 */
class ScoreSchedule : public Backend {
 public:
  /**
   * @if chinese
   * @brief 初始化，配置参数，初始化由Schedule::backend指定的子后端。
   *
   * @param Schedule::backend 子后端，默认为 RangeMerger.
   * 凑batch之外的功能由其实现。与他复合形成了完整的单节点调度功能。
   * @param batching_timeout 凑batch超时时间，默认为0（毫秒）。
   * @param node_name 节点名称。默认为空，用于输出debug信息。
   *
   * @note Schedule::backend的max()大于1时，将启动凑batch线程 @ref
   * ScoreSchedule::run.
   *
   * @else
   * @endif
   */
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config);

  /**
   * @return UINT32_MAX.
   */
  virtual uint32_t max() const { return UINT32_MAX; };

  /**
   *
   * @param[in] TASK_EVENT_KEY
   * 可选，std::shared_ptr<SimpleEvents>，存在的话代表异步请求；否则将构造该项，调用主引擎后阻塞到调用结束。
   * @note
   * Schedule::backend的max()大于1时，会将数据放入到凑batch队列中由
   * @ref ScoreSchedule::run
   * 线程处理。否则直接交给Schedule::backend处理。
   * @warning 一旦自己或者组batch的线程将数据交由 Schedule::backend
   * 处理， 将失去数据的所有权，不可再操作此数据（dict），
   * 相关迭代器也可能失效；等到event被通知后，才重新获得数据的所有权(读写权)。
   */
  void forward(const std::vector<dict>& raw_inputs);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  ~ScoreSchedule();
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
 protected:
#endif

  /**
   * @brief 以超时时间 batching_timeout，
   * 以及数量Schedule::backend->max()为目标进行凑batch.
   * 凑出的一组数据打包执行 Schedule::backend->forward。
   *
   */
  virtual void run();

 private:
  uint32_t max_batch_size_{1};
  std::thread thread_;
  ThreadSafeSortList<float, dict> input_;
  float batching_timeout_;

  std::unique_ptr<Params> params_;
  std::string node_name_;
  std::unique_ptr<Backend> backend_;

  std::atomic_bool bThreadInited_{false};

  std::exception_ptr init_eptr_;

  float number_factor_;

  // std::atomic<unsigned> count_{0};

  std::shared_ptr<RuningState> runing_state_;
};
}  // namespace ipipe
