#pragma once
// #include "threadsafe_queue.hpp"
#include "omniback/builtin/basic_backends.hpp"
#include "omniback/helper/threadsafe_queue.hpp"

namespace omniback {

class RestartEvent : public DependencyV0 {
  void pre_init(const std::unordered_map<string, string>& config, const dict&)
      override final;
  void custom_forward_with_dep(
      const std::vector<dict>& inputs,
      Backend& dependency) override final;

  ~RestartEvent() {
    bInited_.store(false);
    for (auto& one_thread : threads_)
      if (one_thread.joinable()) {
        one_thread.join();
      }
  }

  void task_loop(std::size_t thread_index, ThreadSafeQueue<dict>* pqueue);

 protected:
  struct Stack {
    dict input_data;
    std::shared_ptr<Event> input_event;
    std::size_t task_queue_index = 0;
    std::string request_id;
    Backend* dependency = nullptr;
  };

  void on_start_node(
      dict tmp_data,
      std::size_t task_queue_index,
      Backend& dependency);
  void on_finish_node(dict tmp_data);

 private:
 private:
  std::vector<std::unique_ptr<ThreadSafeQueue<dict>>> task_queues_;
  std::vector<std::thread> threads_;
  std::atomic_bool bInited_{true};
};
} // namespace omniback