#include "hami/core/event.hpp"
#include "hami/schedule/event_guard.hpp"
#include "hami/helper/macro.h"
#include "hami/core/string.hpp"
#include "hami/core/reflect.h"
#include "hami/core/queue.hpp"
#include "BS_thread_pool.hpp"
namespace hami {

void EventGuard::custom_forward_with_dep(const std::vector<dict>& input_output,
                                         Backend* dependency) {
    std::vector<dict> evented_data;
    std::vector<dict> data;

    for (auto item : input_output) {
        if (item->find(TASK_EVENT_KEY) == item->end()) {
            data.push_back(item);
        } else {
            evented_data.push_back(item);
        }
    }
    if (data.empty()) {
        dependency->forward(evented_data);
    } else {
        std::vector<std::shared_ptr<Event>> events(data.size());
        std::generate_n(events.begin(), data.size(),
                        []() { return std::make_shared<Event>(); });
        for (size_t i = 0; i < data.size(); i++) {
            (*data[i])[TASK_EVENT_KEY] = events[i];
        }

        dependency->forward(input_output);
        // parse exception
        std::vector<std::exception_ptr> exceps;
        for (size_t i = 0; i < events.size(); i++) {
            auto expcep = events[i]->wait_and_get_except();
            if (expcep) exceps.push_back(expcep);
            data[i]->erase(TASK_EVENT_KEY);
        }
        if (exceps.size() == 1) {
            std::rethrow_exception(exceps[0]);
        } else if (exceps.size() > 1) {
            std::string msg;
            for (auto& e : exceps) {
                try {
                    std::rethrow_exception(e);
                } catch (const std::exception& e) {
                    msg += std::string("; ") + e.what();
                }
            }
            throw std::runtime_error(msg);
        }
    }
}

HAMI_REGISTER(Backend, EventGuard, "EventGuard,EventGuard");

class ThreadPoolExecutor : public Backend {
   private:
    void impl_init(const std::unordered_map<std::string, std::string>& config,
                   const dict&) override final {
        str::try_update<size_t>(config, "max_workers", max_workers_);
        // pool_ = std::make_unique<BS::thread_pool>(max_workers_);
    }

    void impl_forward_with_dep(const std::vector<dict>& input,
                               Backend* dep) override {
        // while(true){
        //     auto data = queue_->get();
        //     dep->forward({data});
        // }
    }

   protected:
    // std::string target_name_;
    Queue* queue_{nullptr};
    // size_t queue_max_ = 0;
    std::unique_ptr<BS::thread_pool<>> pool_;
    size_t max_workers_{0};
};

}  // namespace hami