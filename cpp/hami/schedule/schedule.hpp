#pragma once

#include <thread>
// #include "threadsafe_queue.hpp"
#include "hami/builtin/basic_backends.hpp"
// #include  "hami/helper/threadsafe_queue.hpp"
#include "hami/core/helper.hpp"
#include "hami/helper/threadsafe_queue.hpp"
#include "hami/helper/threadsafe_sized_queue.hpp"
#include "hami/schedule/schedule_states.hpp"
#include "hami/core/queue.hpp"
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
    if (instances_state_->query_avaliable(req_size, timeout, false)) {
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
  struct CBProtocol {
    // enum struct Action { Stop, Cancel };
    id_type req_id;
    int32_t req_tokens{0};
    int32_t new_tokens{0};
    int32_t max_new_tokens{0};
    int32_t max_tokens{0};
    bool stop{false}; // error, cancel
  };

 private:
  void impl_init(
      const std::unordered_map<string, string>& params,
      const dict& options) override {}
  void impl_forward(const std::vector<dict>& io) override;
};
// #  CBStatus Loop(src_queue)[ContiguousBatching] TASK_MSG_KEY
// xieyi

} // namespace hami