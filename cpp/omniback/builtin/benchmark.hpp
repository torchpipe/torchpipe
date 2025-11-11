
#pragma once

#include <condition_variable>
#include <string>
#include <thread>
#include <vector>
#include "omniback/core/backend.hpp"
#include "omniback/core/queue.hpp"
namespace omniback {
constexpr auto TASK_ProfileState_KEY = "_ProfileState";
struct ProfileState {
  size_t client_index;
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;

  std::chrono::steady_clock::time_point arrive_time;

  std::vector<dict> data;

  std::exception_ptr exception;
};

/**
 * @brief Benchmark is a backend that can be used to benchmark the performance
 * Benchmark[optional[queue_name]]
 * if not specified, the default queue will be used.
 */
class Benchmark : public Backend {
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override;
  void impl_forward_with_dep(
      const std::vector<dict>& input,
      Backend& dependency) override;
  ~Benchmark() {
    bInited_.store(false);
    task_cv_.notify_all();

    for (auto& item : threads_) {
      if (item.joinable()) {
        item.join();
      }
    }
  }

 private:
  void run(size_t client_index);
  std::unordered_map<std::string, std::string> get_output(
      std::exception_ptr& first_exception);

 private:
  size_t num_clients_ = 10;
  size_t request_batch_ = 1;
  size_t total_number_ = 10000;
  size_t num_warm_up_ = 20;
  Queue* target_queue_{nullptr};
  // Queue* src_queue_{nullptr};

 private:
  std::unique_ptr<queue::ThreadSafeQueue<std::shared_ptr<ProfileState>>>
      inputs_;
  queue::ThreadSafeQueue<std::shared_ptr<ProfileState>> outputs_;

  std::function<void()> warm_up_task_;
  std::function<void(size_t)> main_task_;
  std::vector<std::thread> threads_;

  std::mutex warm_up_mtx_;
  std::condition_variable task_cv_;

  std::atomic_bool bNoNewData_{false};
  std::atomic<size_t> warm_up_finished_{0};
  // std::atomic<size_t> main_task_finished_{0};

  std::atomic_bool bInited_{false};
};

} // namespace omniback