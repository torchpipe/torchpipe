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

class Loop : public Backend {
 private:
  void impl_init(const std::unordered_map<string, string>& config, const dict&)
      override;
  void impl_forward(const std::vector<dict>& input) override;
  virtual void run();
  void impl_inject_dependency(Backend* dep) override {
    if (!injected_dependency_)
      injected_dependency_ = dep;
    else {
      injected_dependency_->inject_dependency(dep);
    }
  }

  ~Loop() {
    bInited_.store(false);

    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  void impl_forward_sync(const std::vector<dict>& input);

 private:
  std::atomic_bool bInited_{false};
  std::thread thread_;
  Queue* src_queue_{nullptr};

  [[nodiscard]] size_t impl_max() const override {
    return injected_dependency_->max();
  };
  [[nodiscard]] size_t impl_min() const override {
    return injected_dependency_->min();
  };

 protected:
  Backend* injected_dependency_{nullptr};
  std::string node_name_;
  int max_{std::numeric_limits<int>::max()};
  int timeout_{0};
};

class Batching : public Dependency {
 private:
  void impl_init(const std::unordered_map<string, string>& config, const dict&)
      override final;
  void impl_forward_with_dep(const std::vector<dict>& input, Backend* dep)
      override final;
  virtual void run();
  void impl_inject_dependency(Backend* dependency) override;

 public:
  ~Batching() {
    bInited_.store(false);
    input_queue_.notify_all();
    instances_state_->notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  bool try_forward(
      const std::vector<dict>& input_output,
      size_t req_size,
      size_t timeout) {
    if (instances_state_->query_available(req_size, timeout, false)) {
      injected_dependency_->forward(input_output);
      return true;
    }

    return false;
  }

  std::atomic_bool bInited_{false};
  int batching_timeout_ = 0;
  std::thread thread_;
  ThreadSafeSizedQueue<dict> input_queue_;
  std::shared_ptr<InstancesState> instances_state_;
  std::string node_name_;
};

class BackgroundThread : public Backend {
 private:
  void impl_init(const std::unordered_map<string, string>& config, const dict&)
      override final;
  void impl_forward(const std::vector<dict>& inputs) override final;

  virtual void run();
  [[nodiscard]] size_t impl_max() const override final {
    return dependency_->max();
  }
  [[nodiscard]] size_t impl_min() const override final {
    return dependency_->min();
  }

 public:
  ~BackgroundThread() {
    bInited_.store(false);
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  std::atomic_bool bInited_{false};
  std::atomic_bool bStoped_{false};
  std::thread thread_;
  ThreadSafeQueue<std::vector<dict>> batched_queue_;
  std::string dependency_name_;
  std::unique_ptr<Backend> dependency_;
  std::exception_ptr init_eptr_;
  std::function<void(void)> init_task_;
  // std::function<void(void)> init_task_;
  // void forward_task(const std::vector<dict>& inputs);
};

class InstanceDispatcher : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final;
  virtual void impl_forward(const std::vector<dict>& inputs) override;

  [[nodiscard]] size_t impl_max() const override final {
    return max_;
  }
  [[nodiscard]] size_t impl_min() const override final {
    return min_;
  }

 private:
  void update_min_max(const std::vector<Backend*>& deps);

  size_t max_{1};
  size_t min_{std::numeric_limits<std::size_t>::max()};

 protected:
  std::vector<Backend*> base_dependencies_;
  // std::unique_ptr<InstancesState> instances_state_;
  std::shared_ptr<InstancesState> instances_state_;
};

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

// #  CBStatus Loop(src_queue)[ContiguousBatching] TASK_MSG_KEY

class FakeInstance : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& dict_config) override;

  void impl_forward(const std::vector<dict>& ios);

  [[nodiscard]] size_t impl_max() const override {
    return max_;
  }

  [[nodiscard]] size_t impl_min() const override {
    return min_;
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  // 先销毁消费线程， 确保销毁顺序
  ~FakeInstance() {
    backends_.clear();
  }
#endif
 private:
  // from
  // https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
  template <typename T>
  std::vector<std::size_t> sort_indexes(const std::vector<T>& v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
      return v[i1]->max() <= v[i2]->max();
    });

    return idx;
  }
  int get_best_match(std::size_t size_of_input) {
    for (auto item : sorted_max_) {
      if (size_of_input >= backends_[item]->min() &&
          (size_of_input <= backends_[item]->max())) {
        return item;
      }
    }
    return -1;
  }

 private:
  std::vector<std::unique_ptr<Backend>> backends_;

  size_t fake_instance_num_;
  size_t max_{0};
  size_t min_{0};

  std::vector<std::size_t> sorted_max_;

  // std::unordered_map<std::string, std::string> config_;
};
} // namespace hami