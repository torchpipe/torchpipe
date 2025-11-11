#pragma once
#include <thread>

#include "omniback/core/backend.hpp"
#include "omniback/core/helper.hpp"
#include "omniback/core/queue.hpp"
namespace omniback {
// class Queue;

// init = List[QueueBackend[register_name, optional[target_name]]]
class QueueBackend : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict&) override final;

  void impl_inject_dependency(Backend* dep) override;

  void impl_forward(const std::vector<dict>& input) override final {
    event_guard_forward(
        [this](const std::vector<dict>& data) {
          queue_->puts(data);
          // for (auto& item : data) {
          //     queue_->put(item);
          // }
        },
        input);
  }

  void run();

 public:
  ~QueueBackend();

 protected:
  Queue* queue_{nullptr};
  Backend* target_backend_{nullptr};

 private:
  virtual void pre_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict&);
  // virtual Backend* get_target_backend();
  std::unique_ptr<Queue> owned_queue_{std::make_unique<Queue>()};

  std::atomic_bool bInited_{false};
  std::thread thread_;
  std::string register_name_;
  std::string target_name_;
};

class Send : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict&) override final;

  void impl_forward(const std::vector<dict>& input) override;

 protected:
  // std::string target_name_;
  Queue* queue_{nullptr};
  size_t queue_max_{std::numeric_limits<size_t>::max()};
};

class Observer : public Send {
 private:
  void impl_forward(const std::vector<dict>& input) override final;
};

class Recv : public QueueBackend {
 private:
  void pre_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict&) override final;
};

} // namespace omniback