
#include <algorithm>
// #include <execution>

#include "EventLoop.hpp"
#include "base_logging.hpp"
#include "event.hpp"
// #include "threadsafe_kv_storage.hpp"
#include "exception.hpp"
namespace ipipe {
bool EventLoop::init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
  std::string class_name_backend = IPIPE_GET_REGISTER_NAME(Backend, EventLoop, this);
  class_name_backend += "::backend";
  params_ = std::unique_ptr<Params>(
      new Params({{class_name_backend, "PipelineV3"}, {"continue", TASK_RESTART_KEY}}, {}, {}, {}));
  if (!params_->init(config)) return false;
  continue_ = params_->at("continue");
  constexpr auto N = 2;  // M:N

  for (std::size_t i = 0; i < N; ++i) {
    task_queues_.push_back(std::make_unique<ThreadSafeQueue<dict>>());
    threads_.emplace_back(std::thread(&EventLoop::task_loop, this, i, task_queues_[i].get()));
  }

  auto iter = dict_config->find("backend");
  IPIPE_ASSERT(iter == dict_config->end());
  if (iter != dict_config->end()) {
    backend_ = any_cast<Backend*>(iter->second);
    assert(backend_);
  } else {
    owned_backend_ =
        std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at(class_name_backend)));
    if (!owned_backend_ || !owned_backend_->init(config, dict_config)) return false;
    backend_ = owned_backend_.get();
  }

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
    IPIPE_ASSERT(item->find(TASK_EVENT_KEY) != item->end());
    // if (item->find(TASK_EVENT_KEY) == item->end()) {
    //   (*item)[TASK_EVENT_KEY] = make_event();
    // }

    IPIPE_ASSERT(item->find("request_id") != item->end(), "request_id is needed for EventLoop");
    task_queues_[std::rand() % task_queues_size]->Push(item);
  }
}

void EventLoop::on_start_node(dict tmp_data, std::size_t task_queue_index) {
  std::shared_ptr<EventLoop::Stack> pstack = std::make_shared<EventLoop::Stack>();

  pstack->task_queue_index = task_queue_index;
  pstack->input_event = any_cast<std::shared_ptr<SimpleEvents>>(tmp_data->at(TASK_EVENT_KEY));
  pstack->request_id = any_cast<std::string>(tmp_data->at("request_id"));

  // auto& kv_storage = ThreadSafeKVStorage::getInstance();
  // kv_storage.set(pstack->request_id, TASK_EVENT_KEY, pstack->input_event);

  auto current_event = make_event();
  (*tmp_data)[TASK_EVENT_KEY] = current_event;

  pstack->input_data = tmp_data;
  //   tmp_data->erase(iter);

  //   auto curr_event = make_event();
  auto* local_queue = task_queues_[pstack->task_queue_index].get();  // may not exist
  current_event->set_final_callback([local_queue, tmp_data, pstack]() {
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

  IPIPE_ASSERT(iter != tmp_data->end());

  std::shared_ptr<SimpleEvents> pre_event = any_cast<std::shared_ptr<SimpleEvents>>(iter->second);
  IPIPE_ASSERT((pre_event));
  while (bInited_.load()) {
    // call back called before notify
    if (pre_event->WaitFinish(50)) break;
    SPDLOG_WARN("wait need to much time");
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
    // auto& storage = ThreadSafeKVStorage::getInstance().get(pstack->request_id);
    for (auto iter = tmp_data->begin(); iter != tmp_data->end(); ++iter) {
      if (ignore_keys.find(iter->first) == ignore_keys.end()) {
        // SPDLOG_DEBUG("set storage: {}", iter->first);
        if (pstack->input_data != tmp_data) pstack->input_data->insert(*iter);
        // storage.set(iter->first, iter->second);
      }
    }
    pstack->input_data->erase(TASK_STACK_KEY);

    pstack->input_event->notify_all();
    return;
  } else {
    auto iter = tmp_data->find(TASK_RESTART_KEY);

    std::string restart_node_name = any_cast<std::string>(iter->second);
    tmp_data->erase(iter);
    (*tmp_data)["node_name"] = restart_node_name;
    SPDLOG_DEBUG("RESTART: " + restart_node_name);

    // new iteration
    auto curr_event = make_event();
    (*tmp_data)[TASK_EVENT_KEY] = curr_event;
    tmp_data->erase(TASK_STACK_KEY);

    //   auto curr_data = pstack->input_data;
    auto* local_queue = task_queues_[pstack->task_queue_index].get();
    curr_event->set_final_callback([local_queue, tmp_data, pstack]() {
      (*tmp_data)[TASK_STACK_KEY] = pstack;
      local_queue->Push(tmp_data);
    });

    backend_->forward({tmp_data});
  }
}

IPIPE_REGISTER(Backend, EventLoop, "EventLoop,Ring");
IPIPE_SET_DEFAULT_FRONTEND("Ring,EventLoop", "EnsureInputHasEvent");

class EnsureInputHasEvent : public Backend {
 private:
  std::unique_ptr<Params> params_;
  std::unique_ptr<Backend> backend_;

 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override {
    params_ = std::unique_ptr<Params>(
        new Params({{"EnsureInputHasEvent::backend", "EventLoop"}}, {}, {}, {}));
    if (!params_->init(config)) return false;
    backend_ = std::unique_ptr<Backend>(
        IPIPE_CREATE(Backend, params_->at("EnsureInputHasEvent::backend")));
    if (!backend_ || !backend_->init(config, dict_config)) return false;

    return true;
  }

  void forward(const std::vector<dict>& inputs) override {
    std::vector<dict> evented_data;
    std::vector<dict> data;

    for (auto item : inputs) {
      if (item->find(TASK_EVENT_KEY) == item->end()) {
        data.push_back(item);
      } else {
        evented_data.push_back(item);
      }
    }
    if (!evented_data.empty()) {
      backend_->forward(evented_data);
    }
    if (!data.empty()) {
      std::vector<std::shared_ptr<SimpleEvents>> events(data.size());
      std::generate_n(events.begin(), data.size(),
                      []() { return std::make_shared<SimpleEvents>(); });
      for (size_t i = 0; i < data.size(); i++) {
        (*data[i])[TASK_EVENT_KEY] = events[i];
      }

      backend_->forward(data);
      // parse exception
      std::vector<std::exception_ptr> exceps;
      for (size_t i = 0; i < events.size(); i++) {
        auto expcep = events[i]->WaitAndGetExcept();
        if (expcep) exceps.push_back(expcep);
        data[i]->erase(TASK_EVENT_KEY);
      }
      if (exceps.size() == 1) {
        std::rethrow_exception(exceps[0]);
      } else if (exceps.size() > 1) {
        // throw runtime_error with concated message
        std::string msg;
        for (auto& e : exceps) {
          try {
            std::rethrow_exception(e);
          } catch (const std::exception& e) {
            msg += e.what();
          }
        }
        throw std::runtime_error(msg);
      }
    }
  }
};

IPIPE_REGISTER(Backend, EnsureInputHasEvent, "EnsureInputHasEvent");

}  // namespace ipipe