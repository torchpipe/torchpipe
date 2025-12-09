#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// InstanceDispatcher, Batching (动态dependency)
// forward instance="node_name.0"
namespace omniback {
// IoCV0[SharedInstancesState,InstanceDispatcher,Batching;DI_v0[Batching,
// InstanceDispatcher]]
constexpr auto TASK_RESOURCE_STATE_KEY = "_resource_state";
constexpr auto TASK_REQUEST_STATE_KEY = "_request_state";

// class ResourceState {
//    public:
//     // @return handle
//     virtual std::optional<size_t> wait_available(size_t req_size,
//                                                  size_t timeout,
//                                                  bool erase_available) = 0;
// };

class InstancesState {
 public:
  void add_and_set_range(size_t handle, size_t min_value, size_t max_value);

  void add(size_t handle) {
    {
      std::unique_lock<std::mutex> lock(mtx_);
      available_instances_.insert(handle);
    }

    cv_.notify_all();
  }

  size_t running_intance_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return instances_.size() - available_instances_.size();
  }

  std::optional<size_t> query_available(
      size_t req_size,
      size_t timeout,
      bool lock_queried,
      std::string node_name = "");

  bool wait_for(size_t req_size, size_t timeout);

  void remove_lock(size_t handle) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      locked_available_instances_.erase(handle);
      available_instances_.insert(handle);
    }
    cv_.notify_all();
  }

  void notify_all() {
    cv_.notify_all();
  }

 private:
  mutable std::mutex mtx_;

  std::condition_variable cv_;

  std::unordered_set<size_t> available_instances_;
  std::unordered_set<size_t> locked_available_instances_;
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
        OMNI_ASSERT(all_status_.count(in_status),
                    "Invalid status: " + in_status);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            status_ = in_status;
        }
        cv_.notify_all();
    }

    bool is_status(const std::string& in_status) const {
        OMNI_ASSERT(all_status_.count(in_status),
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
} // namespace omniback