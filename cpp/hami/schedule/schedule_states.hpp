#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <unordered_set>
#include <unordered_map>
#include <optional>

// InstanceDispatcher, Batching (动态dependency)
// forward instance="node_name.0"
namespace hami {
// IoCV0[SharedInstancesState,InstanceDispatcher,Batching;DI_v0[Batching,
// InstanceDispatcher]]
constexpr auto TASK_RESOURCE_STATE_KEY = "_resource_state";
constexpr auto TASK_REQUEST_STATE_KEY = "_request_state";

// class ResourceState {
//    public:
//     // @return handle
//     virtual std::optional<size_t> wait_avaliable(size_t req_size,
//                                                  size_t timeout,
//                                                  bool erase_avaliable) = 0;
// };

class InstancesState {
 public:
  void add_and_set_range(size_t handle, size_t min_value, size_t max_value);

  void add(size_t handle) {
    {
      std::unique_lock<std::mutex> lock(mtx_);
      avaliable_instances_.insert(handle);
    }

    cv_.notify_all();
  }

  std::optional<size_t> query_avaliable(
      size_t req_size,
      size_t timeout,
      bool lock_queried);

  void remove_lock(size_t handle) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      locked_avaliable_instances_.erase(handle);
      avaliable_instances_.insert(handle);
    }
    cv_.notify_all();
  }

  void notify_all() {
    cv_.notify_all();
  }

 private:
  mutable std::mutex mtx_;

  std::condition_variable cv_;

  std::unordered_set<size_t> avaliable_instances_;
  std::unordered_set<size_t> locked_avaliable_instances_;
  std::unordered_map<size_t, std::pair<size_t, size_t>> instances_;
};

#if 0
class Status {
   public:
    // enum class StatusType { RUNNING, STOPPED, PAUSED, ERROR };
    const std::unordered_set<std::string> ALL_STATUS = {
        "RUNNING", "STOPPED", "PAUSED", "ERROR", "WAITING"};
    explicit Status(const std::unordered_set<std::string>& status)
        : all_status_(status) {}
    void set_status(const std::string& in_status) {
        HAMI_ASSERT(all_status_.count(in_status),
                    "Invalid status: " + in_status);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            status_ = in_status;
        }
        cv_.notify_all();
    }

    bool is_status(const std::string& in_status) const {
        HAMI_ASSERT(all_status_.count(in_status),
                    "Invalid status: " + in_status);
        std::lock_guard<std::mutex> lock(mtx_);
        return status_ == in_status;
    }
    // std::string get_status() const {
    //     std::lock_guard<std::mutex> lock(mtx_);
    //     return status_;
    // }
    // const std::unordered_map<StatusType, std::string> status_map = {
    //     {StatusType::RUNNING, "RUNNING"},
    //     {StatusType::STOPPED, "STOPPED"},
    //     {StatusType::PAUSED, "PAUSED"},
    //     {StatusType::ERROR, "ERROR"}};
   private:
    std::string status_;
    const std::unordered_set<std::string>& all_status_;
    mutable std::mutex mtx_;

    std::condition_variable cv_;
};
#endif
} // namespace hami