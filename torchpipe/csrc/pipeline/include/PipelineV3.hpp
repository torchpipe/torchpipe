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

#include <memory>
#include <set>
#include <unordered_map>

#include "Backend.hpp"
#include "dict.hpp"
#include "config_parser.hpp"
#include "event.hpp"
#include "filter.hpp"
#include "LogicalGraph.hpp"

#include "Stack.hpp"

#include "threadsafe_queue.hpp"
namespace ipipe {

class PipelineV3 : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict) override;

  void forward(const std::vector<dict>& inputs) override;

  uint32_t max() const override {
    return UINT32_MAX;  // UINT_MAX
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  ~PipelineV3() {
    bInited_.store(false);
    for (auto& one_thread : threads_)
      if (one_thread.joinable()) {
        one_thread.join();
      }
  }
#endif

 private:
  void on_finish_node(dict tmp_data);
  void on_start_node(dict tmp_data, std::size_t task_queue_index);

  void task_loop(std::size_t thread_index, ThreadSafeQueue<dict>* pqueue);

  bool init(mapmap config);
  void on_map_filter_data(std::string node_name, std::shared_ptr<Stack> pstack);
  void on_filter_data(std::string node_name, std::shared_ptr<Stack> pstack, dict data);
  void forward(dict input, std::size_t task_queues_index, std::shared_ptr<SimpleEvents> event);

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> config_;

  std::unordered_map<std::string, std::unique_ptr<Filter>> filters_;

  std::vector<std::shared_ptr<ThreadSafeQueue<dict>>> task_queues_;

  std::unordered_map<std::unordered_map<std::string, std::string>*, Stack> stacks_;  // 协程栈
  std::vector<std::thread> threads_;
  std::atomic_bool bInited_{false};

  std::shared_ptr<LogicalGraph> graph_;

  std::unordered_map<std::string, std::set<std::string>> split_nodes_;
  std::unordered_map<std::string, std::set<std::string>> reduce_nodes_;
  dict dict_config_;

  std::unique_ptr<Backend> physical_view_;
};

}  // namespace ipipe