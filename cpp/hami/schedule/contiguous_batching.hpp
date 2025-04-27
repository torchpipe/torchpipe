#pragma once

#include <thread>
// #include "threadsafe_queue.hpp"
#include "hami/builtin/basic_backends.hpp"
// #include  "hami/helper/threadsafe_queue.hpp"
#include "hami/core/helper.hpp"
#include "hami/core/queue.hpp"
#include "hami/helper/threadsafe_queue.hpp"
#include "hami/helper/threadsafe_sized_queue.hpp"
#include "hami/schedule/schedule_states.hpp"
#include "hami/builtin/page_table.hpp"
namespace hami {

// IOC[instancedd, executor]
class ContiguousBatching : public Backend {
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

    float time{0};
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

} // namespace hami