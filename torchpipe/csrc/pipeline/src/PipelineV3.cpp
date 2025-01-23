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

#include "PipelineV3.hpp"

#include <cassert>
#include <string>
#include <vector>
#include <memory>

#include "config_parser.hpp"

#include <algorithm>
#include <cstdlib>
#include <future>
#include "base_logging.hpp"
#include "dep_sort_stl.h"
#include "event.hpp"
#include "filter.hpp"
#include "reflect.h"
#include "base_logging.hpp"
#include "PhysicalView.hpp"
#include "exception.hpp"
#include "params.hpp"
namespace ipipe {

inline bool all_processed(const std::set<std::string>& nodes,
                          const std::unordered_map<std::string, dict>& processed) {
  for (const auto& node : nodes) {
    if (processed.find(node) == processed.end()) {
      return false;
    }
  }
  return true;
}

bool PipelineV3::init(const std::unordered_map<std::string, std::string>& config,
                      dict dict_config) {
  physical_view_ = std::make_unique<PhysicalView>();

  IPIPE_ASSERT(physical_view_ && physical_view_->init(config, dict_config));

  dict_config_ = dict_config;
  mapmap global_config;
  if (dict_config && dict_config->find("config") != dict_config->end()) {
    global_config = any_cast<mapmap>((*dict_config)["config"]);
    return init(global_config);
  } else if (!config.empty()) {
    global_config[TASK_DEFAULT_NAME_KEY] = config;
    return init(global_config);
  }
  return false;
}

bool PipelineV3::init(mapmap config) {
  if (config.size() == 1 && config.find("global") != config.end()) {
    config[TASK_DEFAULT_NAME_KEY] = config["global"];
  }
  graph_ = std::make_shared<LogicalGraph>();

  config.erase("global");
  for (const auto& item : config) {
    auto iter_next = item.second.find("next");
    if (iter_next != item.second.end()) {
      auto vec_of_next = str_split(iter_next->second, ',');
      for (const auto& item_next : vec_of_next) {
        if (config.count(item_next) == 0) {
          SPDLOG_ERROR("`{}`'s next `{}` is invalid.", item.first, item_next);
          return false;
        }
      }
      graph_->add(item.first, vec_of_next);

    } else {
      graph_->add(item.first);
    }
  }
  if (!graph_->finalize()) {
    SPDLOG_ERROR("unable to finalize the graph.");
    return false;
  }

  const auto& sortted = graph_->get_sortted();
  const auto& root_nodes = graph_->get_roots();

  // handle map reduce
  for (const auto& item : config) {
    auto iter_map_reduce = item.second.find("map");
    if (iter_map_reduce != item.second.end()) {
      graph_->set_map_reduce(item.first, iter_map_reduce->second, false);
    } else {
    }
  }

  if (!graph_->check_map()) {
    SPDLOG_ERROR("check map failed.");
    return false;
  }

  // 打印信息

  std::stringstream ss;
  ss << "Graph information:"
     << "\n";
  for (auto node_n : sortted) {
    std::string back_end_name = brackets_combine(config[node_n]);

    ss << node_n << "(" << back_end_name << ")"
       << "\n";
  }

  SPDLOG_INFO(colored(ss.str()));

  // build filter and backend
  for (auto& iter_next : config) {
    // 开始处理filter： 用于逻辑控制功能
    auto iter_filter = iter_next.second.find("filter");

    std::unique_ptr<Filter> filter;

    if (iter_filter != iter_next.second.end()) {
      if (graph_->is_root(iter_next.first)) {
        SPDLOG_ERROR("root node {} should not have filter", iter_next.first);
        return false;
      }
      brackets_split(iter_next.second, "filter");
      filters_[iter_next.first] =
          std::unique_ptr<Filter>(IPIPE_CREATE(Filter, iter_next.second["filter"]));
    } else {
      filters_[iter_next.first] = std::unique_ptr<Filter>(
          IPIPE_CREATE(Filter, graph_->get_default_filter(iter_next.first)));
    }
    if (!filters_[iter_next.first] || !filters_[iter_next.first]->init(iter_next.second, nullptr)) {
      SPDLOG_ERROR("{}: filter init failed", iter_next.first);
      return false;
    }
  }

  std::swap(config_, config);
  if (root_nodes.empty()) {
    SPDLOG_ERROR("empty root nodes");
    return false;
  }
  bInited_.store(true);
  constexpr auto N = 2;  // M:N Coroutine
  task_queues_ = std::vector<std::shared_ptr<ThreadSafeQueue<dict>>>();

  // task_queues_和threads_分开处理
  for (std::size_t i = 0; i < N; ++i) {
    task_queues_.push_back(std::make_shared<ThreadSafeQueue<dict>>());
    threads_.emplace_back(std::thread(&PipelineV3::task_loop, this, i, task_queues_[i].get()));
  }

  return true;  // todo
}

void PipelineV3::task_loop(std::size_t thread_index, ThreadSafeQueue<dict>* pqueue) {
  while (bInited_.load()) {
    dict tmp_data = nullptr;
    if (pqueue->WaitForPop(tmp_data, 100)) {
      if (tmp_data->find(TASK_STACK_KEY) == tmp_data->end()) {
        on_start_node(tmp_data, thread_index);
      } else {
        on_finish_node(tmp_data);
      }
    }
  }

  SPDLOG_INFO("task_loop exit({})", thread_index);
  return;
}

void PipelineV3::on_finish_node(dict tmp_data) {
  std::string node_name = any_cast<std::string>((*tmp_data)["node_name"]);

  auto iter = tmp_data->find(TASK_STACK_KEY);
  assert(iter != tmp_data->end());
  std::shared_ptr<Stack> pstack = any_cast<std::shared_ptr<Stack>>(iter->second);
  assert(pstack);

  iter = tmp_data->find(TASK_EVENT_KEY);

  if (iter != tmp_data->end()) {
    std::shared_ptr<SimpleEvents> curr_event =
        any_cast<std::shared_ptr<SimpleEvents>>(iter->second);
    IPIPE_ASSERT((curr_event));
    while (bInited_.load()) {
      // call back called before notify
      if (curr_event->WaitFinish(50)) break;
    }

    tmp_data->erase(TASK_EVENT_KEY);
    if (curr_event->has_exception()) {
      if (!pstack->exception)
        pstack->exception = insert_exception(
            curr_event->reset_exception(),
            (__func__ + std::string(": while processing node `" + node_name + '`')));
      else {
        curr_event->reset_exception();
      }
    }
  } else {
    // assert(false);
    SPDLOG_DEBUG("PipelineV3: on finish node: no event found");
  }
  if (!pstack->valid(node_name)) {
    SPDLOG_ERROR("PipelineV3: on finish node: invalid stack. node_name = {}", node_name);
    // return;
  } else {
    pstack->update_processed(node_name, tmp_data);
  }

  if (pstack->exception) {  // todo check
    if (pstack->allStopped()) {
      // no task is running.
      pstack->input_data->erase(TASK_STACK_KEY);
      pstack->input_data->erase(TASK_RESULT_KEY);

      if (pstack->input_event)  // todo: check
      {
        (*pstack->input_data)[TASK_EVENT_KEY] = pstack->input_event;
      }

      pstack->input_event->set_exception_and_notify_all(pstack->exception);
      pstack->clear();
    }
    return;
  }

  if (pstack->waiting_nodes.empty() && pstack->allStopped()) {
    // finished
    assert(pstack->allFinished());
    if (pstack->end_nodes.size() == 1) {
      if (pstack->input_data != tmp_data) {
        std::swap(*pstack->input_data, *tmp_data);
      }
      pstack->input_data->erase(TASK_STACK_KEY);
      (*pstack->input_data)[TASK_EVENT_KEY] = pstack->input_event;
      pstack->input_event->notify_all();

      pstack->clear();
    } else if (pstack->end_nodes.size() > 1) {
      pstack->input_data->erase(TASK_RESULT_KEY);
      (*pstack->input_data)[TASK_STACK_KEY] = pstack;
      (*pstack->input_data)[TASK_EVENT_KEY] = pstack->input_event;
      pstack->input_event->notify_all();

    } else {
      assert(false);
    }

  } else {
    const auto copy_waiting_nodes = pstack->waiting_nodes;
    for (const auto& waiting_node : copy_waiting_nodes) {  // 同一线程调度，不会出现问题。
      if (pstack->waiting_nodes.count(waiting_node) == 0) continue;  // 可能中途被更新
      const auto& pre = pstack->graph->get_previous(waiting_node);
      assert(!pre.empty());
      if (all_processed(pre, pstack->processed)) {
        pstack->waiting_nodes.erase(waiting_node);
        pstack->non_waiting_nodes.insert(waiting_node);

        on_map_filter_data(waiting_node, pstack);

        if (pstack->exception) break;
      }
    }
  }
  return;
}

void PipelineV3::on_start_node(dict tmp_data, std::size_t task_queue_index) {
  std::string node_name = any_cast<std::string>((*tmp_data)["node_name"]);
  std::shared_ptr<Stack> pstack = std::make_shared<Stack>();

  pstack->task_queue_index = task_queue_index;
  auto iter = tmp_data->find(TASK_EVENT_KEY);
  assert(iter != tmp_data->end());
  pstack->input_event = any_cast<std::shared_ptr<SimpleEvents>>(iter->second);
  assert(pstack->input_event->valid());
  pstack->input_data = tmp_data;
  tmp_data->erase(iter);
  pstack->start_node = node_name;

  try {
    const auto& root_nodes = graph_->get_roots();
    const auto& sortted = graph_->get_sortted();
    if (std::find(sortted.begin(), sortted.end(), node_name) == sortted.end()) {
      throw std::out_of_range(node_name + " isn't in the graph.");
    }
    // create a new stack
    // 作为输入， 需要确保 TASK_EVENT_KEY 一定在
    pstack->graph = graph_;
    if (root_nodes.count(node_name) == 0) {
      pstack->graph = pstack->graph->as_root(node_name);
      // filters_[node_name] = std::unique_ptr<Filter>(IPIPE_CREATE(Filter, "Run"));
      IPIPE_ASSERT(pstack->graph != nullptr, node_name + " can not be setted as root node.");
    }

    // assert(root_nodes.count(node_name) != 0);

    pstack->non_waiting_nodes.insert(node_name);
    pstack->update_status();

    on_filter_data(node_name, pstack, pstack->input_data);
  } catch (const std::exception& e) {
    auto curr_data = make_dict(node_name);
    assert(curr_data);
    (*curr_data)[TASK_STACK_KEY] = pstack;
    pstack->exception = insert_exception(
        e.what(), (__func__ + std::string(": while processing node `" + node_name + '`')));

    task_queues_[task_queue_index]->Push(curr_data);
    return;
  }
}

void PipelineV3::forward(const std::vector<dict>& inputs) {
  std::size_t num_event = 0;
  for (auto& item : inputs) {
    if (item->find(TASK_EVENT_KEY) != item->end()) {
      num_event++;
    }
  }

  static const auto task_queues_size = task_queues_.size();
  std::size_t task_queues_index = std::rand() % task_queues_size;
  if (num_event == inputs.size()) {
    for (auto& item : inputs) {
      auto iter = item->find(TASK_EVENT_KEY);

      /**There is something wrong. But we choose not to throw here.*/
      std::shared_ptr<SimpleEvents> curr_event;
      try {
        curr_event = any_cast<std::shared_ptr<SimpleEvents>>(iter->second);
        if (!curr_event || !curr_event->valid()) {
          throw std::invalid_argument("input event is invalid");
        }
      } catch (std::exception& e) {
        SPDLOG_ERROR(e.what());
        item->erase(TASK_RESULT_KEY);
        continue;
      }

      forward(item, task_queues_index, curr_event);
    }
  } else if (num_event == 0) {
    auto ev = make_event(inputs.size());
    for (auto& item : inputs) {
      (*item)[TASK_EVENT_KEY] = ev;
    }
    for (auto& item : inputs) forward(item, task_queues_index, ev);

    auto exc = ev->WaitAndGetExcept();

    for (auto& item : inputs) {
      item->erase(TASK_EVENT_KEY);
      assert(item->find(TASK_STACK_KEY) == item->end());
    }

    if (exc) {
      for (auto& item : inputs) {
        item->erase(TASK_RESULT_KEY);
      }
      std::rethrow_exception(exc);
    }

  } else {
    std::vector<dict> async_inputs;
    std::vector<dict> sync_inputs;
    for (auto& item : inputs) {
      if (item->find(TASK_EVENT_KEY) != item->end()) {
        async_inputs.push_back(item);
      } else {
        sync_inputs.push_back(item);
      }
    }
    forward(async_inputs);  // never throw
    forward(sync_inputs);   // may throw

    return;
  }
  return;
}

void PipelineV3::forward(dict input, std::size_t task_queues_index,
                         std::shared_ptr<SimpleEvents> curr_event) {
  assert(input && curr_event && curr_event->valid());
  try {
    const auto& root_nodes = graph_->get_roots();

    std::string node_name;
    auto iter = input->find("node_name");
    if (iter == input->end()) {
      if (root_nodes.size() > 1) {
        std::ostringstream errr;
        for (const auto& node : root_nodes) {
          errr << node << " ";  // 假设每个节点有一个 name() 方法
        }
        throw std::out_of_range("node_name not set. existing: " + errr.str());
      }
      node_name = *root_nodes.begin();
    } else {
      node_name = any_cast<std::string>(iter->second);
      if (root_nodes.find(node_name) == root_nodes.end()) {
        // throw std::out_of_range(node_name + " isn't one of root nodes.");
        SPDLOG_DEBUG(node_name + " isn't one of root nodes. treat it as root node.");
        // graph_->as_root(node_name);
        if (!graph_->is_valid(node_name)) {
          throw std::out_of_range(node_name + " isn't in the graph.");
        }
      }
    }
    (*input)["node_name"] = node_name;
    task_queues_[task_queues_index]->Push(input);
  } catch (...) {
    input->erase(TASK_RESULT_KEY);
    curr_event->set_exception_and_notify_all(std::current_exception());
  }
}

void PipelineV3::on_filter_data(std::string node_name, std::shared_ptr<Stack> pstack,
                                dict curr_data) {
  auto filter_result = Filter::status::Error;
  if (pstack->graph->is_root(node_name)) {
    filter_result = Filter::status::Run;
  } else {
    auto iter_filter = filters_.find(node_name);  // must have one
    assert(iter_filter != filters_.end());
    filter_result = iter_filter->second->forward(curr_data);
  }

  pstack->set_filter_status(node_name, filter_result);
  switch (filter_result) {
    case Filter::status::Error:
      SPDLOG_DEBUG("filter Filter::status::Error. node name is {}", node_name);
      throw std::runtime_error("filter error");
    case Filter::status::Run: {
      auto curr_event = make_event();
      std::weak_ptr<ThreadSafeQueue<dict>> local_queue =
          task_queues_[pstack->task_queue_index];  // may not exist
      curr_event->set_final_callback([local_queue, curr_data, pstack]() {
        auto shared_q = local_queue.lock();
        if (shared_q) {
          (*curr_data)[TASK_STACK_KEY] = pstack;
          shared_q->Push(curr_data);
        }
      });

      (*curr_data)[TASK_EVENT_KEY] = curr_event;
      curr_data->erase(TASK_RESULT_KEY);  // todo: move this line to Schedule
      curr_data->erase(TASK_STACK_KEY);
      assert(curr_data->find(TASK_EVENT_KEY) != curr_data->end());
      physical_view_->forward({curr_data});
      return;
    }
    case Filter::status::Break:
      pstack->stop(node_name);
      break;
    case Filter::status::Skip:
      break;
    case Filter::status::SerialSkip:
      pstack->serial_skip(node_name);
      break;
    case Filter::status::SubGraphSkip:
      pstack->graph_skip(node_name);
      break;
  }
  (*curr_data)[TASK_STACK_KEY] = pstack;
  task_queues_[pstack->task_queue_index]->Push(curr_data);
}

void PipelineV3::on_map_filter_data(std::string node_name, std::shared_ptr<Stack> pstack) {
  auto input_event = pstack->input_event;
  assert(input_event != nullptr);

  dict curr_data;

  try {
    curr_data = pstack->get_mapped_previous(
        node_name);  // pstack->graph->get_mapped_previous(node_name, pstack->processed);

    IPIPE_ASSERT(curr_data);
    on_filter_data(node_name, pstack, curr_data);
  } catch (...) {
    if (!curr_data) curr_data = make_dict(node_name);
    assert(curr_data);
    (*curr_data)[TASK_STACK_KEY] = pstack;
    pstack->exception = std::current_exception();
    if (pstack->exception)  // SPDLOG_ERROR("{}: {}", node_name, "on_map_filter_data failed");
      pstack->exception =
          insert_exception(pstack->exception, " while processing node " + node_name);
    task_queues_[pstack->task_queue_index]->Push(curr_data);
    return;
  }
  return;
}

IPIPE_REGISTER(Backend, PipelineV3, "PipelineV3");
}  // namespace ipipe