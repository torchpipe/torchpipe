#include "omniback/core/event.hpp"
#include "omniback/helper/base_logging.hpp"

#include "omniback/builtin/aspect.hpp"
#include "omniback/builtin/proxy.hpp"
#include "omniback/core/task_keys.hpp"
#include "omniback/schedule/restart.hpp"

namespace omniback {

void RestartEvent::pre_init(
    const std::unordered_map<string, string>& config,
    const dict& kwargs) {
  constexpr auto N = 1; // M:N         set to one if 需要严格保持先来的先处理

  // TODO unique  EventBus
  for (std::size_t i = 0; i < N; ++i) {
    task_queues_.push_back(std::make_unique<ThreadSafeQueue<dict>>());
    threads_.emplace_back(
        std::thread(&RestartEvent::task_loop, this, i, task_queues_[i].get()));
  }
}

void RestartEvent::task_loop(
    std::size_t thread_index,
    ThreadSafeQueue<dict>* pqueue) {
  while (bInited_.load()) {
    dict tmp_data = nullptr;
    if (pqueue->wait_pop(tmp_data, 100)) {
      OMNI_ASSERT(tmp_data->find(TASK_STACK_KEY) != tmp_data->end());
      if (tmp_data->find(TASK_STACK_KEY) == tmp_data->end()) {
        // on_start_node(tmp_data, thread_index);
      } else {
        on_finish_node(tmp_data);
      }
    }
  }

  SPDLOG_INFO("RestartEvent task_loop exit.");
  return;
};

void RestartEvent::custom_forward_with_dep(
    const std::vector<dict>& inputs,
    Backend& dependency) {
  const size_t queue_index = std::rand() % task_queues_.size();

  for (auto& item : inputs) {
    OMNI_ASSERT(item->find(TASK_EVENT_KEY) != item->end());
    OMNI_ASSERT(item->find(TASK_STACK_KEY) == item->end());
  }
  // task_queues_[queue_index]->push(inputs);
  for (const auto& item : inputs) {
    on_start_node(item, queue_index, dependency);
  }
}

void RestartEvent::on_start_node(
    dict tmp_data,
    std::size_t task_queue_index,
    Backend& dependency) {
  std::shared_ptr<RestartEvent::Stack> pstack =
      std::make_shared<RestartEvent::Stack>();

  pstack->dependency = &dependency;
  pstack->task_queue_index = task_queue_index;
  pstack->input_event =
      any_cast<Event>(tmp_data->at(TASK_EVENT_KEY));

  auto current_event = Event();
  (*tmp_data)[TASK_EVENT_KEY] = current_event;

  pstack->input_data = tmp_data;

  auto* local_queue = task_queues_[pstack->task_queue_index].get();
  current_event->set_final_callback([local_queue, tmp_data, pstack]() {
    (*tmp_data)[TASK_STACK_KEY] = pstack;
    local_queue->push(tmp_data);
  });

  dependency.forward({tmp_data});
}

void RestartEvent::on_finish_node(dict tmp_data) {
  auto iter = tmp_data->find(TASK_STACK_KEY);
  OMNI_FATAL_ASSERT(iter != tmp_data->end());
  std::shared_ptr<Stack> pstack =
      any_cast<std::shared_ptr<Stack>>(iter->second);
  OMNI_FATAL_ASSERT(pstack);

  iter = tmp_data->find(TASK_EVENT_KEY);

  OMNI_FATAL_ASSERT(iter != tmp_data->end());
  Event pre_event =
      any_cast<Event>(iter->second);
  // OMNI_FATAL_ASSERT((pre_event));
  //   tmp_data->erase(iter);

  // TODO the following is useless and should be removed
  while (bInited_.load()) {
    if (pre_event->wait_finish(50))
      break;
    SPDLOG_WARN("wait need to much time");
  }

  if (pre_event->has_exception()) {
    pstack->input_data->erase(TASK_STACK_KEY);
    pstack->input_data->erase(TASK_RESULT_KEY);

    pstack->input_data->insert({TASK_EVENT_KEY, pstack->input_event});
    pstack->input_event->set_exception_and_notify_all(
        pre_event->reset_exception());
    return;
  } else if (
      tmp_data->find(TASK_RESTART_KEY) == tmp_data->end() ||
      tmp_data->find(TASK_RESULT_KEY) == tmp_data->end()) { //
    static const std::unordered_set<std::string> ignore_keys = {
        TASK_STACK_KEY, TASK_EVENT_KEY, TASK_DATA_KEY};
    for (auto iter = tmp_data->begin(); iter != tmp_data->end(); ++iter) {
      if (ignore_keys.count(iter->first) == 0) {
        if (pstack->input_data != tmp_data)
          pstack->input_data->insert(*iter);
      }
    }
    pstack->input_data->erase(TASK_STACK_KEY);
    // pstack->input_data->erase(TASK_RESULT_KEY);

    pstack->input_data->insert({TASK_EVENT_KEY, pstack->input_event});
    pstack->input_event->notify_all();
    return;
  } else {
    auto iter = tmp_data->find(TASK_RESTART_KEY);

    std::string restart_node_name = any_cast<std::string>(iter->second);
    tmp_data->erase(iter);
    (*tmp_data)[TASK_NODE_NAME_KEY] = restart_node_name;
    SPDLOG_DEBUG("RESTART to: " + restart_node_name);
    iter = tmp_data->find(TASK_RESULT_KEY);
    (*tmp_data)[TASK_DATA_KEY] = iter->second;

    // new iteration
    auto curr_event = Event();
    (*tmp_data)[TASK_EVENT_KEY] = curr_event;
    tmp_data->erase(TASK_STACK_KEY);

    auto* local_queue = task_queues_[pstack->task_queue_index].get();
    curr_event->set_final_callback([local_queue, tmp_data, pstack]() {
      (*tmp_data)[TASK_STACK_KEY] = pstack;
      local_queue->push(tmp_data);
    });

    pstack->dependency->forward({tmp_data});
  }
}
OMNI_REGISTER(Backend, RestartEvent);

OMNI_PROXY_WITH_DEPENDENCY(Restart, "Aspect[EventGuard,RestartEvent]");

} // namespace omniback