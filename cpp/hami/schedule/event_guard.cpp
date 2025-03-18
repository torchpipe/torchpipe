#include "hami/core/event.hpp"
#include "hami/schedule/event_guard.hpp"
#include "hami/helper/macro.h"
#include "hami/core/string.hpp"
#include "hami/core/reflect.h"
#include "hami/core/queue.hpp"
#include "hami/builtin/control_plane.hpp"
#include "hami/core/parser.hpp"
#include "BS_thread_pool.hpp"
#include "hami/schedule/schedule_states.hpp"
#include "hami/core/task_keys.hpp"
#include "hami/helper/base_logging.hpp"
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

HAMI_REGISTER(Backend, EventGuard, "EventGuard");

class ThreadPoolExecutor : public DynamicDependency {
   private:
    void impl_init(const std::unordered_map<std::string, std::string>& params,
                   const dict& options) override final {
        auto args_kwargs =
            meta::get_args_kwargs(this, "ThreadPoolExecutor", params);

        str::try_update<size_t>(args_kwargs.second, "max_workers",
                                max_workers_);

        if (max_workers_ == 0) {
            max_workers_ = std::thread::hardware_concurrency();
        } else {
            max_workers_ += 1;
        }
        pool_ = std::make_unique<BS::thread_pool<>>(max_workers_);

        // get the src queue
        // HAMI_ASSERT(!args_kwargs.first.empty(),
        //             "ThreadPoolExecutor: args_kwargs is empty");
        // queue_ = HAMI_INSTANCE_GET(Queue, args_kwargs.first[0]);
        // HAMI_ASSERT(queue_ != nullptr);
        std::string queue_tag;
        str::try_update(args_kwargs.second, "out", queue_tag);

        target_queue_ = &default_queue(queue_tag);

        if (max_workers_ == 0) {
            max_workers_ = pool_->get_thread_count();
        }
    }

    // [[nodiscard]] size_t impl_max() const { return max_workers_; }
    void impl_forward_with_dep(const std::vector<dict>& input,
                               Backend* dep) override {
        (void)pool_->submit_task(
            [this, input, dep]() { impl_forward_with_dep_async(input, dep); });
    }
    void impl_forward_with_dep_async(const std::vector<dict>& input,
                                     Backend* dep) {
        HAMI_ASSERT(input.size() == 1);
        Queue* queue = dict_get<Queue*>(input[0], TASK_DATA_KEY);
        HAMI_ASSERT(queue && pool_);
        std::vector<std::future<void>> futures;
        std::size_t index = 0;
        do {
            auto [data, len] = queue->try_get(std::chrono::milliseconds(500));
            // SPDLOG_INFO("queue get {}", len);
            if (!data) continue;
            // std::future<void> future =
            (void)pool_->submit_task([this, dep, data, queue]() {
                try {
                    dep->forward({*data});
                } catch (...) {
                    // queue->set_error();
                    (*(*data))["exception"] = std::current_exception();
                }

                target_queue_->put(*data);
            });
            // futures.push_back(std::move(future));
            if (index++ % (100) == 0) {
                SPDLOG_INFO("processing {}/{}", index, queue->size());
            }
        } while (queue->status() == Queue::Status::RUNNING && alive_.load());

        // for (auto& future : futures) {
        //     future.get();
        // }
    }

   protected:
    // std::string target_name_;
    // Queue* queue_{nullptr};
    // size_t queue_max_ = 0;
    std::unique_ptr<BS::thread_pool<>> pool_;
    size_t max_workers_{0};
    // Status* state_;
    Queue* target_queue_{nullptr};
    std::atomic_bool alive_{true};

   public:
    ~ThreadPoolExecutor() {
        alive_.store(false);
        pool_.release();
    }
};

HAMI_REGISTER_BACKEND(ThreadPoolExecutor);

}  // namespace hami