#pragma once
#include "hami/core/queue.hpp"
#include "hami/core/backend.hpp"

namespace hami {
// class Queue;

// init = List[QueueBackend[register_name, optional[target_name]]]
class QueueBackend : public Backend {
   public:
    void init(const std::unordered_map<std::string, std::string>& config,
              const dict&) override final;

    void inject_dependency(Backend* dep) override;

    void forward(const std::vector<dict>& input) override final {
        for (auto& item : input) {
            queue_->put(item);
        }
    }

    void run();

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
   public:
    void init(const std::unordered_map<std::string, std::string>& config,
              const dict&) override final;

    void forward(const std::vector<dict>& input) override final {
        for (auto& item : input) {
            queue_->put(item);
        }
    }

   private:
    std::string target_name_;
    Queue* queue_{nullptr};
};

class Recv : public QueueBackend {
   private:
    void pre_init(const std::unordered_map<std::string, std::string>& config,
                  const dict&) override final;
};

}  // namespace hami