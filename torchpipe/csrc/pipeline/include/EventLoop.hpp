#include <memory>
#include <vector>
#include <atomic>
#include <thread>

#include "threadsafe_queue.hpp"
#include "Backend.hpp"
#include "event.hpp"
#include "params.hpp"
#pragma once
namespace ipipe {
class EventLoop : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict) override;
  void forward(const std::vector<dict>& inputs) override;

  ~EventLoop() {
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
    std::shared_ptr<SimpleEvents> input_event;
    std::size_t task_queue_index = 0;
    std::string request_id;
  };

  void on_start_node(dict tmp_data, std::size_t task_queue_index);
  void on_finish_node(dict tmp_data);

 private:
  // void forward(dict input, std::size_t task_queues_index, std::shared_ptr<SimpleEvents>
  // curr_event);

 private:
  std::vector<std::unique_ptr<ThreadSafeQueue<dict>>> task_queues_;
  std::vector<std::thread> threads_;
  std::atomic_bool bInited_{true};
  std::unique_ptr<Backend> owned_backend_;
  Backend* backend_;
  std::unique_ptr<Params> params_;
  std::string continue_;
};
}  // namespace ipipe
