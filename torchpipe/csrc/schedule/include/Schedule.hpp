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
#include "Instances.hpp"
#include "params.hpp"

#include "threadsafe_queue.hpp"
namespace ipipe {

class Schedule : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config);

  virtual uint32_t max() const { return UINT32_MAX; };

  /**
   *
   * @param[in] TASK_EVENT_KEY
   * 可选，std::shared_ptr<SimpleEvents>，存在的话代表异步请求；否则将构造该项，调用主引擎后阻塞到调用结束。
   * @note Schedule::backend的max()大于1时，会将数据放入到凑batch队列中由
   * @ref Schedule::run 线程处理。否则直接交给Schedule::backend处理。
   * @warning 一旦自己或者组batch的线程将数据交由 Schedule::backend 处理，
   * 将失去数据的所有权，不可再操作此数据（dict），
   * 相关迭代器也可能失效；等到event被通知后，才重新获得数据的所有权(读写权)。
   */
  void forward(const std::vector<dict>& raw_inputs);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  ~Schedule();
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
 private:
#endif

  /**
   * @brief 以超时时间 batching_timeout，
   * 以及数量Schedule::backend->max()为目标进行凑batch.
   * 凑出的一组数据打包执行 Schedule::backend->forward。
   *
   */
  void run();

 private:
  void async_forward(const std::vector<dict>& raw_inputs);
  uint32_t max_batch_size_{1};
  std::thread thread_;
  ThreadSafeQueue<dict> input_queue_;
  float batching_timeout_;

  std::unique_ptr<Params> params_;
  std::string node_name_;
  std::unique_ptr<Backend> backend_;

  std::atomic_bool bInited_{false};
};
}  // namespace ipipe
