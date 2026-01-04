#include "omniback/schedule/event_guard.hpp"

#include "BS_thread_pool.hpp"
#include "omniback/builtin/control_plane.hpp"
#include "omniback/core/event.hpp"
#include "omniback/core/parser.hpp"
#include "omniback/core/queue.hpp"
#include "omniback/core/reflect.h"
#include "omniback/core/string.hpp"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
#include "omniback/schedule/schedule_states.hpp"
namespace omniback {

void EventGuard::custom_forward_with_dep(
    const std::vector<dict>& input_output,
    Backend& dependency) {
  std::vector<dict> evented_data;
  std::vector<dict> data;

  for (auto item : input_output) {
    if (item->find(TASK_EVENT_KEY) == item->end()) {
      data.push_back(item);
    } else {
      evented_data.push_back(item);
    }
  }
  if (data.empty()) {
    // SPDLOG_INFO("EVENT_GUARD: all has event. size = {}",
    // evented_data.size());
    dependency.forward(evented_data);
  } else {
    std::vector<Event> events(data.size());
    std::generate_n(events.begin(), data.size(), []() {
      return Event();
    });
    for (size_t i = 0; i < data.size(); i++) {
      (*data[i])[TASK_EVENT_KEY] = events[i];
    }

    dependency.forward(input_output);
    // parse exception
    std::vector<std::exception_ptr> exceps;
    for (size_t i = 0; i < events.size(); i++) {
      auto expcep = events[i]->wait_and_get_except();
      if (expcep)
        exceps.push_back(expcep);
      data[i]->erase(TASK_EVENT_KEY);
    }
    if (exceps.size() == 1) {
      std::rethrow_exception(exceps[0]);
    } else if (exceps.size() > 1) {
      std::string msg;
      for (auto& e : exceps) {
        try {
          std::rethrow_exception(e);
        } catch (const std::exception& e) {
          msg += std::string("; ") + e.what();
        }
      }
      throw std::runtime_error(msg);
    }
  }
}

OMNI_REGISTER(Backend, EventGuard, "EventGuard");

class ThreadPoolExecutor : public Dependency {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override final {
    auto args_kwargs =
        parser_v2::get_args_kwargs(this, "ThreadPoolExecutor", params);

    str::try_update<size_t>(args_kwargs.second, "max_workers", max_workers_);

    if (max_workers_ == 0) {
      max_workers_ = std::thread::hardware_concurrency();
    } else {
      max_workers_ += 1;
    }
    pool_ = std::make_unique<BS::thread_pool<>>(max_workers_);
    SPDLOG_INFO("max_workers = {} ", max_workers_);

    // get the src queue
    // OMNI_ASSERT(!args_kwargs.first.empty(),
    //             "ThreadPoolExecutor: args_kwargs is empty");
    // queue_ = OMNI_INSTANCE_GET(Queue, args_kwargs.first[0]);
    // OMNI_ASSERT(queue_ != nullptr);
    std::string queue_tag;
    str::try_update(args_kwargs.second, "out", queue_tag);

    target_queue_ = &default_queue(queue_tag);

    if (max_workers_ == 0) {
      max_workers_ = pool_->get_thread_count();
    }
  }

  // [[nodiscard]] uint32_t impl_max() const { return max_workers_; }
  void impl_forward_with_dep(const std::vector<dict>& input, Backend& dep)
      override {
    (void)pool_->submit_task(
        [this, input, &dep]() { impl_forward_with_dep_async(input, dep); });
  }
  void impl_forward_with_dep_async(
      const std::vector<dict>& input,
      Backend& dep) {
    OMNI_ASSERT(input.size() == 1);
    Queue* queue = dict_get<Queue*>(input[0], TASK_DATA_KEY);
    OMNI_ASSERT(queue && pool_);

    do {
      // SPDLOG_INFO(" pool queue input : {}", queue->size());
      auto data_opt = queue->try_get<dict>(500);
      // SPDLOG_INFO("queue get {}", len);
      if (!data_opt)
        continue;
      auto data = data_opt.value();
      // SPDLOG_INFO("queue :  {}, {} {}", queue->size(),
      // pool_->get_tasks_total(), pool_->get_tasks_queued());
      pool_->detach_task([this, &dep, data, queue]() {
        try {
          dep.forward({data});
        } catch (...) {
          // queue->set_error();
          (*data)["exception"] = std::current_exception();
        }
        target_queue_->push(data);
      });

      while (!pool_->wait_atmost_queued_tasks_for(
          max_workers_ / 3 + 1, std::chrono::milliseconds(500)))
        ;

    } while (alive_.load());

    while (
        alive_.load() &&
        !pool_->wait_atmost_total_tasks_for(1, std::chrono::milliseconds(500)))
      ;
  }

 protected:
  // std::string target_name_;
  // Queue* queue_{nullptr};
  // size_t queue_max_ = 0;
  std::unique_ptr<BS::thread_pool<>> pool_;
  size_t max_workers_{0};
  // Status* state_;
  Queue* target_queue_{nullptr};
  std::atomic_bool alive_{true};
  std::atomic<size_t> index_{0};

  // private:
  //     std::condition_variable need_new_cv_;
  //     std::mutex need_new_mutex_;

 public:
  ~ThreadPoolExecutor() {
    alive_.store(false);
    pool_.release();
  }
};

OMNI_REGISTER_BACKEND(ThreadPoolExecutor);

} // namespace omniback