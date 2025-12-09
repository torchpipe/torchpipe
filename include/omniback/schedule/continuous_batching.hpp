#pragma once

#include <thread>
// #include "threadsafe_queue.hpp"
#include "omniback/builtin/basic_backends.hpp"
// #include  "omniback/helper/threadsafe_queue.hpp"
#include "omniback/builtin/page_table.hpp"
#include "omniback/core/helper.hpp"
#include "omniback/core/queue.hpp"
#include "omniback/helper/threadsafe_queue.hpp"
#include "omniback/helper/threadsafe_sized_queue.hpp"
#include "omniback/helper/threadsafe_unordered_map.hpp"
#include "omniback/schedule/schedule_states.hpp"

namespace omniback {

class PlainContinuousBatching : public Backend {
 private:
  struct TaskInfo {
    std::string id;
    // static_cast<float>(std::chrono::duration<double>(
    //                        std::chrono::system_clock::now().time_since_epoch())
    //                        .count())
    dict data;
    float time{0};
    std::shared_ptr<Event> event;
    int loop_index = 0;
    int delay = 0;
  };
  void impl_init(
      const std::unordered_map<string, string>& params,
      const dict& options) override;
  void impl_forward(const std::vector<dict>& io) override;
  bool all_received() {
    const auto keys = cached_data_.keys();
    for (const auto& item : keys) {
      if (receiving_data_.find(item) == receiving_data_.end()) {
        return false;
      }
    }
    return true;
  }

 private:
  //  int max_{std::numeric_limits<int>::max()};
  Backend* dependency_{nullptr};
  std::unordered_map<std::string, TaskInfo> receiving_data_;
  std::mutex mutex_;

 private:
  void task_loop();

 private:
  // struct Group{
  //   std::thread thread;
  //   std::unordered_map<std::string, TaskInfo> cached_data;
  // };

  Queue* src_queue_;
  std::thread thread_;
  threadsafe_unordered_map<std::string, TaskInfo> cached_data_;
  std::atomic_bool bInited_{true};

  ~PlainContinuousBatching() {
    bInited_.store(false);
    if (thread_.joinable()) {
      thread_.join();
    }
  }
};

// IOC[instancedd, executor]
class ContinuousBatching : public Backend {
 public:
  struct BatchInfo {
    // enum struct Action { Stop, Cancel };
    id_type req_id;

    int req_tokens{0};
    int context_length{0};
    int max_tokens{0};

    // bool stop{false}; // error, cancel //stop by error or cancel
    bool finish{false};

    int new_tokens{0};

    size_t new_page_needed{0};
    bool running = false;
    dict data;

    std::shared_ptr<Event> event;

    double time{0};
    // std::string req_type = "prefill";
  };

 private:
  void impl_init(
      const std::unordered_map<string, string>& params,
      const dict& options) override;
  void impl_forward(const std::vector<dict>& io) override;
  void impl_forward_handle_except(
      const std::vector<dict>& ios,
      const std::vector<id_type>& ids);
  Backend* dependency_{nullptr};
  void parser_message(
      const std::shared_ptr<TypedDict>& msg,
      BatchInfo& protocol);

 private:
  std::pair<std::vector<id_type>, std::unordered_map<id_type, std::string>>
  get_activated_ids();

  void stable_sort_by_time(std::vector<id_type>& ids) {
    std::stable_sort(
        ids.begin(), ids.end(), [this](const id_type& a, const id_type& b) {
          return req_status_.at(a).time <= req_status_.at(b).time;
        });
  }

  template <typename T>
  void stable_sort_by_time(std::vector<std::pair<id_type, T>>& ids) {
    std::stable_sort(
        ids.begin(),
        ids.end(),
        [this](const std::pair<id_type, T>& a, const std::pair<id_type, T>& b) {
          return req_status_.at(a.first).time <= req_status_.at(b.first).time;
        });
  }

  std::unordered_map<id_type, BatchInfo> req_status_;
  // std::mutex req_status_mutex_;
  PageTable* page_table_{nullptr};
  int page_size_{0};
  int max_{std::numeric_limits<int>::max()};
  // std::unordered_set<id_type> need_stop_;
  std::unique_ptr<Backend> no_page_table_;
};

} // namespace omniback