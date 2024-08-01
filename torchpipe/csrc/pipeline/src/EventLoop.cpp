#include "EventLoop.hpp"
#include "base_logging.hpp"
#include "event.hpp"
#include "threadsafe_kv_storage.hpp"
#include "exception.hpp"
namespace ipipe {
bool EventLoop::init(const std::unordered_map<std::string, std::string>& config, dict) {
  params_ = std::unique_ptr<Params>(
      new Params({{"EventLoop::backend", "SyncRing"}, {"continue", TASK_RESTART_KEY}}, {}, {}, {}));
  if (!params_->init(config)) return false;
  continue_ = params_->at("continue");
  constexpr auto N = 2;  // M:N

  for (std::size_t i = 0; i < N; ++i) {
    task_queues_.push_back(std::make_unique<ThreadSafeQueue<dict>>());
    threads_.emplace_back(std::thread(&EventLoop::task_loop, this, i, task_queues_[i].get()));
  }

  backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("EventLoop::backend")));
  if (!backend_ || !backend_->init(config, nullptr)) return false;
  return true;
}

void EventLoop::task_loop(std::size_t thread_index, ThreadSafeQueue<dict>* pqueue) {
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
};
void EventLoop::forward(const std::vector<dict>& inputs) {
  static const auto task_queues_size = task_queues_.size();

  for (auto& item : inputs) {
    if (item->find(TASK_EVENT_KEY) == item->end()) {
      (*item)[TASK_EVENT_KEY] = make_event();
    }

    IPIPE_ASSERT(item->find("request_id") != item->end());
    task_queues_[std::rand() % task_queues_size]->Push(item);
  }
}

void EventLoop::on_start_node(dict tmp_data, std::size_t task_queue_index) {
  std::shared_ptr<EventLoop::Stack> pstack = std::make_shared<EventLoop::Stack>();

  pstack->task_queue_index = task_queue_index;
  pstack->input_event = any_cast<std::shared_ptr<SimpleEvents>>(tmp_data->at(TASK_EVENT_KEY));
  pstack->request_id = any_cast<std::string>(tmp_data->at("request_id"));

  auto& kv_storage = ThreadSafeKVStorage::getInstance();
  kv_storage.set(pstack->request_id, TASK_EVENT_KEY, pstack->input_event);

  auto current_event = make_event();
  (*tmp_data)[TASK_EVENT_KEY] = current_event;

  pstack->input_data = tmp_data;
  //   tmp_data->erase(iter);

  //   auto curr_event = make_event();
  auto* local_queue = task_queues_[pstack->task_queue_index].get();  // may not exist
  current_event->add_callback([local_queue, tmp_data, pstack]() {
    (*tmp_data)[TASK_STACK_KEY] = pstack;
    local_queue->Push(tmp_data);
  });

  backend_->forward({tmp_data});
  //   (*curr_data)[TASK_EVENT_KEY] = curr_event;
}

void EventLoop::on_finish_node(dict tmp_data) {
  auto iter = tmp_data->find(TASK_STACK_KEY);
  assert(iter != tmp_data->end());
  std::shared_ptr<Stack> pstack = any_cast<std::shared_ptr<Stack>>(iter->second);
  assert(pstack);

  iter = tmp_data->find(TASK_EVENT_KEY);

  assert(iter != tmp_data->end());

  std::shared_ptr<SimpleEvents> pre_event = any_cast<std::shared_ptr<SimpleEvents>>(iter->second);
  IPIPE_ASSERT((pre_event));
  while (bInited_.load()) {
    // call back called before notify
    if (pre_event->WaitFinish(50)) break;
  }

  //   tmp_data->erase(TASK_EVENT_KEY);

  if (pre_event->has_exception()) {
    pstack->input_data->erase(TASK_STACK_KEY);
    pstack->input_data->erase(TASK_RESULT_KEY);

    pstack->input_event->set_exception_and_notify_all(pre_event->reset_exception());

    // pstack->input_event->set_exception(curr_event->get_exception());
    return;
  } else if (tmp_data->find(continue_) == tmp_data->end()) {
    static const std::set<std::string> ignore_keys = {TASK_STACK_KEY, TASK_EVENT_KEY,
                                                      TASK_DATA_KEY};
    auto& storage = ThreadSafeKVStorage::getInstance().get(pstack->request_id);
    for (auto iter = tmp_data->begin(); iter != tmp_data->end(); ++iter) {
      if (ignore_keys.find(iter->first) == ignore_keys.end()) {
        pstack->input_data->insert(*iter);
        storage.set(iter->first, iter->second);
      }
    }
    pstack->input_data->erase(TASK_STACK_KEY);

    pstack->input_event->notify_all();
    return;
  }

  // new iteration
  auto curr_event = make_event();
  (*tmp_data)[TASK_EVENT_KEY] = curr_event;
  tmp_data->erase(TASK_STACK_KEY);

  //   auto curr_data = pstack->input_data;
  auto* local_queue = task_queues_[pstack->task_queue_index].get();
  curr_event->add_callback([local_queue, tmp_data, pstack]() {
    (*tmp_data)[TASK_STACK_KEY] = pstack;
    local_queue->Push(tmp_data);
  });

  backend_->forward({tmp_data});
}
}  // namespace ipipe