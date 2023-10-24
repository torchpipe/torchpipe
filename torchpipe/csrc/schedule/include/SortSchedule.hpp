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

#include <memory>
#include <string>
#include <vector>
#include "Backend.hpp"
#include "dict.hpp"
#include "params.hpp"
#include "reflect.h"
#include "state_event.hpp"
#include "threadsafe_list.hpp"
#include "time_utils.hpp"

namespace ipipe {

/**
 * @brief 实现单节点调度，凑batch的功能。
 * 实际默认调用后端链：`SortSchedule[Instances[InstanceHandler[backend]]]`，
 * 其中后三个部分可分别通过 `SortSchedule::backend, Instances::backend backend`
 * 参数修改。
 */
class SortSchedule : public Backend {
 public:
  /**
   * @if chinese
   * @brief 初始化，配置参数，初始化由Schedule::backend指定的子后端。
   *
   * @param SortSchedule::backend 子后端，默认为 Instances.
   * 凑batch之外的功能由其实现。与他复合形成了完整的单节点调度功能。
   * @param batching_timeout 凑batch超时时间，默认为0（毫秒）。
   * @param number_factor 如果需要取max_num的 数据，超时时间内最大允许
   * number_factor * max_num个数据堆积， 以便进行择优。 默认1.5.
   * @param node_name 节点名称。默认为空，用于输出debug信息。
   *
   * @note SortSchedule::backend的max()大于1时，将启动凑batch线程 @ref
   * SortSchedule::run : 以超时时间 batching_timeout，
   * 以及 SortSchedule::backend->max()为目标进行凑batch.
   * 凑出的一组数据打包送入 SortSchedule::backend的前向接口。
   *
   * @else
   * @endif
   */
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config);

  virtual uint32_t max() const { return UINT32_MAX; };

  /**
   * @brief
   *
   * @param _sort_score
   * 默认值：程序启动到现在经过的时间的相反数。此输入用于判断输入数据之间的距离，距离近者更有机会凑成批次。
   * 典型的场景：文字识别中，_sort_score设为文字行长宽比。
   */
  void forward(const std::vector<dict>& raw_inputs);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  ~SortSchedule();

 private:
#endif
  /**
   * @brief 以超时时间 batching_timeout，
   * 以及数量SortSchedule::backend->max()为目标进行凑batch.
   * 凑出的一组数据打包执行 Schedule::backend->forward。
   * @remark 如果需要取max_num的 数据，超时时间内最大允许
   * number_factor * max_num个数据堆积，
   * 以便进行择优。择优方法为：首先会选择最老的数据，
   * 然后选择和最老的数据距离最近的足够数量的其他数据。
   * 距离 依照 _sort_score 相对差确定。
   *
   */
  void run();

 private:
  uint32_t max_batch_size_{1};
  std::thread thread_;
  ThreadSafeSortList<float, dict> input_list_;
  float batching_timeout_;
  float number_factor_;

  std::unique_ptr<Params> params_;
  std::string node_name_;
  std::unique_ptr<Backend> backend_;

  std::atomic_bool bInited_{false};

  std::exception_ptr init_eptr_;

  std::shared_ptr<StateEvents> state_;
};
}  // namespace ipipe