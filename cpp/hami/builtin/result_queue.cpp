#include <thread>
#include "hami/builtin/result_queue.hpp"
#include "hami/helper/string.hpp"
#include "hami/core/helper.hpp"
#include "hami/helper/timer.hpp"

namespace hami {

// init = List[QueueBackend[register_name, optional[target_name]]]
void QueueBackend::pre_init(
    const std::unordered_map<std::string, std::string>& config, const dict&) {
    auto dep = get_dependency_name_force(this, config);
    auto iter = dep.find(',');

    if (iter == std::string::npos) {
        register_name_ = dep;
    } else {
        register_name_ = dep.substr(0, iter);
        target_name_ = dep.substr(iter + 1);
    }
    HAMI_ASSERT(!register_name_.empty(),
                "QueueBackend should have register name: "
                "Queue[register_name, optional[target_name]");
    HAMI_INSTANCE_REGISTER(Backend, register_name_, this);
    HAMI_ASSERT(owned_queue_);
    queue_ = owned_queue_.get();
    HAMI_INSTANCE_REGISTER(Queue, register_name_, queue_);

    if (!target_name_.empty()) {
        target_backend_ = HAMI_INSTANCE_GET(Backend, target_name_);
        HAMI_ASSERT(target_backend_);
    }
}

void QueueBackend::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& dict_config) {
    pre_init(config, dict_config);
    if (target_backend_) {
        bInited_ = true;

        thread_ = std::thread(&QueueBackend::run, this);
    }
}

void QueueBackend::impl_inject_dependency(Backend* dep) {
    if (target_backend_) {
        target_backend_->inject_dependency(dep);
    } else {
        target_backend_ = dep;
    }
}

void QueueBackend::run() {
    while (bInited_.load()) {
        auto data = queue_->try_get(SHUTDOWN_TIMEOUT_MS);
        if (!data.first) continue;
        auto io_data = *(data.first);
        assert(io_data);
        (*io_data)[TASK_REQUEST_SIZE_KEY] = data.second;
        target_backend_->forward({io_data});
    }
}

QueueBackend::~QueueBackend() {
    bInited_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

HAMI_REGISTER_BACKEND(QueueBackend, "QueueBackend,AsyncQueue, Queue");

// SrcFromQueue
// init = Register[QueueBackend[x]]
// QueueBackend[register_name, target_name]

// init = List[Send[target_name]]
void Send::impl_init(const std::unordered_map<std::string, std::string>& config,
                     const dict&) {
    auto target_name = get_dependency_name(this, config);
    if (target_name)
        queue_ = HAMI_INSTANCE_GET(Queue, *target_name);
    else {
        queue_ = &(default_queue());
    }
    HAMI_ASSERT(queue_);
}

void Send::impl_forward(const std::vector<dict>& input) {
    queue_->put(input);
    // for (auto& item : input) {
    //     queue_->put(item);
    // }
}

void Observer::impl_forward(const std::vector<dict>& input) {
    for (auto& item : input) {
        (*item)[TASK_RESULT_KEY] = item->at(TASK_DATA_KEY);
        auto new_dict = deep_copy(item);
        queue_->put(new_dict);
    }
}

HAMI_REGISTER_BACKEND(Send);
HAMI_REGISTER_BACKEND(Observer);

// init = List[Recv[register_name_for_src_que, target_backend_name]]
void Recv::pre_init(const std::unordered_map<std::string, std::string>& config,
                    const dict&) {
    auto dep = get_dependency_name_force(this, config);
    auto iter = dep.find(',');

    HAMI_ASSERT(iter != std::string::npos,
                "Usage: Recv[src_queue_name, target_backend_name]");

    auto register_name = dep.substr(0, iter);
    auto target_name = dep.substr(iter + 1);

    HAMI_ASSERT(!register_name.empty() && !target_name.empty(),
                "Recv must have register name and target name");
    queue_ = HAMI_INSTANCE_GET(Queue, register_name);

    target_backend_ = HAMI_INSTANCE_GET(Backend, target_name);
    HAMI_ASSERT(target_backend_);
}

}  // namespace hami