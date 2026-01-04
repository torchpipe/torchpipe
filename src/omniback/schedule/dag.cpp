#include "omniback/schedule/dag.hpp"

#include "omniback/core/helper.hpp"
#include "omniback/core/reflect.h"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/string.hpp"
#include "omniback/helper/threadsafe_queue.hpp"
#include "omniback/helper/timer.hpp"

namespace omniback {
void DagDispatcher::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  OMNI_ASSERT(kwargs, "DagDispatcher: kwargs is required");
  auto iter = kwargs->find(TASK_CONFIG_KEY);
  OMNI_ASSERT(
      iter != kwargs->end(), "DagDispatcher: config not found in kwargs");
  str::mapmap dual_config = any_cast<str::mapmap>(iter->second);

  // per-node settings
  for (const auto& item : dual_config) {
    if (item.first == TASK_GLOBAL_KEY)
      continue;
    Backend* back = OMNI_INSTANCE_GET(Backend, "node." + item.first);
    if (!back) {
      SPDLOG_WARN(
          "DagDispatcher: instance(" + ("node." + item.first) + ") not found");
      continue;
    }
    // OMNI_ASSERT(
    //     back,
    //     "DagDispatcher: instance(" + ("node." + item.first) + ") not found");
    base_dependencies_["node." + item.first] = back;

    auto dag_backend = init_backend(
        "DagProxy",
        {{"DagProxy::dependency", item.first}},
        {},
        "dag." + item.first);
    dag_backend->inject_dependency(this);

    owned_backends_.emplace_back(std::move(dag_backend));
  }

  dag_parser_ = std::make_unique<parser::DagParser>(dual_config);

  // event loop
  constexpr auto N = 1; // M:N         set to one if 需要严格保持先来的先处理

  for (std::size_t i = 0; i < N; ++i) {
    task_queues_.push_back(std::make_unique<ThreadSafeQueue<dict>>());
    threads_.emplace_back(
        std::thread(&DagDispatcher::task_loop, this, i, task_queues_[i].get()));
  }
}

void DagDispatcher::evented_forward(const std::vector<dict>& inputs) {
  const size_t queue_index = std::rand() % task_queues_.size();

  for (auto& item : inputs) {
    OMNI_FATAL_ASSERT(item->find(TASK_STACK_KEY) == item->end());
  }
  on_start_nodes(inputs, queue_index);
}

void DagDispatcher::task_loop(
    std::size_t thread_index,
    ThreadSafeQueue<dict>* pqueue) {
  while (bInited_.load()) {
    dict tmp_data = nullptr;
    if (pqueue->wait_pop(tmp_data, SHUTDOWN_TIMEOUT)) {
      auto iter = tmp_data->find(TASK_STACK_KEY);
      OMNI_ASSERT(iter != tmp_data->end());
      std::shared_ptr<Stack> pstack =
          any_cast<std::shared_ptr<Stack>>(iter->second);
      tmp_data->erase(iter);
      on_finish_node(tmp_data, pstack);
    }
  }

  SPDLOG_INFO("DagDispatcher task_loop exit.");
  return;
};

void DagDispatcher::on_start_nodes(
    const std::vector<dict>& tmp_data,
    std::size_t task_queue_index) {
  std::vector<std::string> node_names;
  std::vector<dict> valid_data;
  for (const auto& io : tmp_data) {
    auto node_name = on_start_node(io, task_queue_index);
    if (!node_name.empty()) {
      node_names.push_back(node_name);
      valid_data.push_back(io);
    }
  }
  if (valid_data.empty())
    return;
  bool all_equal =
      std::adjacent_find(
          node_names.begin(), node_names.end(), std::not_equal_to<>{}) ==
      node_names.end();
  if (all_equal) {
    base_dependencies_.at("node." + node_names.front())->forward(valid_data);
    // SPDLOG_DEBUG(
    //     "all_equal. node name = {}, size = {}",
    //     node_names.front(),
    //     node_names.size());
  } else {
    for (size_t i = 0; i < node_names.size(); ++i) {
      const auto& name = node_names[i];
      if (name.empty())
        continue;
      const auto& data = valid_data[i]; // 取出对应的数据
      base_dependencies_.at("node." + name)->forward({data}); // 单数据转发
    }
  }
}

std::string DagDispatcher::on_start_node(
    const dict& tmp_data,
    std::size_t task_queue_index) {
  std::shared_ptr<DagDispatcher::Stack> pstack =
      std::make_shared<DagDispatcher::Stack>();

  pstack->task_queue_index = task_queue_index;
  pstack->input_event =
      any_cast<Event>(tmp_data->at(TASK_EVENT_KEY));

  // Check if node_name is present in the input
  auto iter_node_name = tmp_data->find(TASK_NODE_NAME_KEY);
  const auto& roots = dag_parser_->get_roots();
  std::string node_name;
  if (iter_node_name == tmp_data->end()) {
    // SPDLOG_INFO("roots =  {}", roots.size());
    if (roots.size() != 1) {
      pstack->input_event->set_exception_and_notify_all(
          std::make_exception_ptr(
              std::runtime_error(
                  "DagDispatcher: `node_name` not found in input. "
                  "Please set it to specify the target node.")));
      return "";
    } else {
      node_name = (*roots.begin());
      // SPDLOG_WARN(
      //     "The parameter node_name is not set, but there are {} root "
      //     "nodes in the graph, making it impossible to determine which "
      //     "one to use. Using the first one: {}",
      //     roots.size(), node_name);
    }
  } else
    node_name = any_cast<std::string>(iter_node_name->second);
  // new event
  auto current_event = Event();
  (*tmp_data)[TASK_EVENT_KEY] = current_event;

  pstack->input_data = tmp_data;
  // node_name = node_name;

  pstack->dag.waiting_nodes = dag_parser_->get_subgraph(node_name);
  pstack->dag.total = pstack->dag.waiting_nodes.size();
  pstack->dag.waiting_nodes.erase(node_name);

  auto* local_queue = task_queues_[pstack->task_queue_index].get();
  current_event->set_final_callback([local_queue, pstack, node_name]() {
    (*pstack->input_data)[TASK_STACK_KEY] = pstack;
    (*pstack->input_data)[TASK_NODE_NAME_KEY] = node_name;
    local_queue->push(pstack->input_data);
  });

  tmp_data->erase(TASK_RESULT_KEY);
  return node_name;
  // base_dependencies_.at("node." + node_name)->forward({tmp_data});
}

void DagDispatcher::on_finish_node(
    dict tmp_data,
    std::shared_ptr<Stack> pstack) {
  assert(pstack);

  std::string node_name =
      any_cast<std::string>(tmp_data->at(TASK_NODE_NAME_KEY));

  // handle event
  auto iter = tmp_data->find(TASK_EVENT_KEY);
  if (iter != tmp_data->end()) {
    Event curr_event =
        any_cast<Event>(iter->second);
    // IPIPE_ASSERT((curr_event));
    while (bInited_.load()) {
      // call back called before notify. May be not needed?
      if (curr_event->wait_finish(SHUTDOWN_TIMEOUT)) {
        break;
      }
      SPDLOG_WARN("wait need to much time.");
    }

    tmp_data->erase(TASK_EVENT_KEY);
    if (curr_event->has_exception()) {
      if (!pstack->exception) {
        pstack->exception = curr_event->reset_exception();
        // SPDLOG_WARN("Exception occurred in node: {}", node_name);
      } else {
        auto tmp_exc = curr_event->reset_exception();
        try {
          std::rethrow_exception(tmp_exc);
        } catch (const std::exception& e) {
          SPDLOG_WARN(
              "Exception in node: {} has been overridden by a new "
              "exception: {}",
              node_name,
              e.what());
        } catch (...) {
          SPDLOG_WARN(
              "Exception in node: {} has been overridden by a new "
              "unknown exception",
              node_name);
        }
      }
    }
  }

  // SPDLOG_DEBUG("processed node_name = {}", node_name);
  pstack->dag.processed[node_name] = tmp_data;

  if (pstack->exception) { // todo check
    if (pstack->dag.waiting_nodes.size() + pstack->dag.processed.size() ==
        pstack->dag.total) {
      // no task is running.
      pstack->input_data->erase(TASK_STACK_KEY);
      pstack->input_data->erase(TASK_RESULT_KEY);

      (*pstack->input_data)[TASK_EVENT_KEY] = pstack->input_event;

      pstack->input_event->set_exception_and_notify_all(pstack->exception);
      clear(pstack.get());
    }
    return;
  }
  // SPDLOG_INFO("Processed : {} {} {} {}", node_name,
  // (bool)pstack->exception,
  //             pstack->dag.waiting_nodes.size(), pstack->dag.total);
  // finished
  if (pstack->dag.processed.size() == pstack->dag.total) {
    assert(pstack->dag.waiting_nodes.empty());

    if (pstack->input_data != tmp_data) {
      std::swap(*pstack->input_data, *tmp_data);
    }

    // SPDLOG_DEBUG("total = {}, waiting: {} has result {}", pstack->dag.total,
    // pstack->dag.waiting_nodes.size(), pstack->input_data->find("result")!=
    // pstack->input_data->end());

    pstack->input_data->erase(TASK_STACK_KEY);
    (*pstack->input_data)[TASK_EVENT_KEY] = pstack->input_event;
    pstack->input_event->notify_all();

    clear(pstack.get());
    return;
  } else {
    // SPDLOG_INFO("Processed : {}. waiting {} total {}", node_name,
    // pstack->dag.waiting_nodes.size(),
    //             pstack->dag.total);

    const auto copy_waiting_nodes = pstack->dag.waiting_nodes;
    for (const auto& waiting_node :
         copy_waiting_nodes) { // 同一线程调度，不会出现问题。
      if (pstack->dag.waiting_nodes.count(waiting_node) == 0)
        continue;

      if (dag_parser_->is_ready(waiting_node, pstack->dag.processed)) {
        pstack->dag.waiting_nodes.erase(waiting_node);

        map_or_filter_data(waiting_node, pstack);

        if (pstack->exception)
          break;
      }
    }
  }
  return;
}

void DagDispatcher::map_or_filter_data(
    std::string node_name,
    std::shared_ptr<Stack> pstack) {
  dict curr_data;

  try {
    curr_data = dag_parser_->prepare_data_from_previous(
        node_name, pstack->dag.processed);
    assert(curr_data);
    (*curr_data)[TASK_NODE_NAME_KEY] = node_name;
    // or filter occur
    if (pstack->dag.processed.size() == pstack->dag.total) {
      curr_data = pstack->dag.processed.at(node_name);
      (*curr_data)[TASK_STACK_KEY] = pstack;
      task_queues_[pstack->task_queue_index]->push(curr_data);
      return;
    }

    execute(node_name, pstack, curr_data);
  } catch (...) {
    if (!curr_data)
      curr_data = make_dict(node_name);
    (*curr_data)[TASK_STACK_KEY] = pstack;
    pstack->exception = std::current_exception();
    task_queues_[pstack->task_queue_index]->push(curr_data);
    return;
  }
  return;
}

void DagDispatcher::execute(
    std::string node_name,
    std::shared_ptr<Stack> pstack,
    dict curr_data) {
  auto curr_event = Event();
  ThreadSafeQueue<dict>* local_queue =
      task_queues_[pstack->task_queue_index].get();
  curr_event->set_final_callback([local_queue, curr_data, pstack, node_name]() {
    (*curr_data)[TASK_STACK_KEY] = pstack;
    (*curr_data)[TASK_NODE_NAME_KEY] = node_name;
    local_queue->push(curr_data);
  });

  (*curr_data)[TASK_EVENT_KEY] = curr_event;
  curr_data->erase(TASK_RESULT_KEY);
  curr_data->erase(TASK_STACK_KEY);

  base_dependencies_["node." + node_name]->forward({curr_data});
  return;
}

class DagProxy : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override {
    auto iter = config.find("DagProxy::dependency");
    OMNI_ASSERT(
        iter != config.end(), "DagProxy: `node_name` not found in config");
    node_name_ = iter->second;
  }
  void impl_forward(const std::vector<dict>& input_output) override {
    for (auto& item : input_output) {
      item->insert_or_assign(TASK_NODE_NAME_KEY, node_name_);
    }
    OMNI_ASSERT(
        injected_dependency_, "DagProxy: injected_dependency_ is nullptr");
    injected_dependency_->forward(input_output);
  }

  void impl_inject_dependency(Backend* dependency) override {
    if (!injected_dependency_) {
      injected_dependency_ = dependency;
    } else {
      injected_dependency_->inject_dependency(dependency);
    }
  }

 private:
  Backend* injected_dependency_{nullptr};
  std::string node_name_;
};
OMNI_REGISTER(Backend, DagProxy);
OMNI_REGISTER(Backend, DagDispatcher);
} // namespace omniback