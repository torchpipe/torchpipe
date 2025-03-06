#include <random>
#include "hami/builtin/benchmark.hpp"
#include "hami/helper/string.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/timer.hpp"

#include "hami/core/event.hpp"
#include "hami/helper/macro.h"
#include "hami/builtin/result_queue.hpp"
#include "hami/core/task_keys.hpp"
#include "hami/core/helper.hpp"

namespace hami {

void Benchmark::init(const std::unordered_map<std::string, std::string>& config,
                     const dict& dict_config) {
    auto dep = get_dependency_name(this, config, "Benchmark");
    if (dep) {
        target_queue_ = HAMI_INSTANCE_GET(Queue, *dep);
    } else {
        target_queue_ = &default_queue();
    }
    HAMI_ASSERT(target_queue_);

    str::try_update(config, "num_clients", num_clients_);
    str::try_update(config, "request_batch", request_batch_);
    str::try_update(config, "total_number", total_number_);
    str::try_update(config, "num_warm_up", num_warm_up_);

    if (total_number_ % request_batch_ != 0) {
        total_number_ += request_batch_ - total_number_ % request_batch_;
        SPDLOG_WARN(
            "total_number is not divisible by request_batch, so we change it "
            "to {}",
            total_number_);
    }

    inputs_ =
        std::make_unique<queue::ThreadSafeQueue<std::shared_ptr<ProfileState>>>(
            num_clients_);

    bInited_.store(true);

    for (size_t i = 0; i < num_clients_; i++) {
        threads_.emplace_back(&Benchmark::run, this, i);
    }

    // stages_.resize(num_clients_, Stage::WaitingForWarmup)
}

void Benchmark::forward(const std::vector<dict>& input, Backend* dependency) {
    HAMI_ASSERT(input.size() > 1 && dependency);

    {
        // warm up
        std::unique_lock<std::mutex> lock(warm_up_mtx_);

        warm_up_task_ = [this, input, dependency]() {
            std::random_device seeder;
            std::mt19937 engine(seeder());
            std::uniform_int_distribution<int> dist(0, input.size() - 1);
            int num_warm_up = num_warm_up_;

            while (num_warm_up-- > 0) {
                std::vector<dict> warm_data;
                for (size_t i = 0; i < request_batch_; ++i) {
                    warm_data.push_back(input[dist(engine)]);
                }
                try {
                    dependency->forward(warm_data);
                } catch (const std::exception& e) {
                    SPDLOG_ERROR("Exception during warm-up forward: {}",
                                 e.what());
                    break;
                }
            }
        };

        main_task_ = [this, dependency](size_t client_index) {
            while (bInited_.load()) {
                auto item = inputs_->try_get(SHUTDOWN_TIMEOUT_MS);
                SPDLOG_INFO("input = {}", inputs_->qsize());
                if (item) {
                    auto data = *item;
                    auto& state = *(data);

                    state.client_index = client_index;
                    state.arrive_time = std::chrono::steady_clock::now();

                    state.start_time = std::chrono::steady_clock::now();
                    std::exception_ptr excep;
                    try {
                        dependency->forward(state.data);
                    } catch (...) {
                        state.exception = (std::current_exception());
                    }
                    state.end_time = std::chrono::steady_clock::now();
                    outputs_.put(data);
                } else if (bNoNewData_.load() && inputs_->empty()) {
                    break;
                }
            }
        };
    }

    task_cv_.notify_all();

    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_int_distribution<int> dist(0, input.size() - 1);

    // generate test data
    size_t req_times = total_number_ / request_batch_;
    // size_t req_times = num_warm_up_ * request_batch_;
    while (req_times-- > 0) {
        auto data = std::make_shared<ProfileState>();
        data->arrive_time = std::chrono::steady_clock::now();

        for (size_t i = 0; i < request_batch_; ++i) {
            data->data.push_back(input[dist(engine)]);
        }
        while (!inputs_->try_put(data,
                                 std::chrono::milliseconds(SHUTDOWN_TIMEOUT))) {
        };
        SPDLOG_INFO("req_times = {} inputs_ size {}", req_times,
                    inputs_->qsize());
    }
    bNoNewData_.store(true);

    auto profile_result = get_output();
    dict data = make_dict();
    (*data)[TASK_DATA_KEY] = profile_result;
    target_queue_->put(data);
    // get outputs
}

void Benchmark::run(size_t client_index) {
    HAMI_ASSERT(bInited_.load());
    {
        std::unique_lock<std::mutex> lock(warm_up_mtx_);
        task_cv_.wait(lock,
                      [this]() { return warm_up_task_ || !bInited_.load(); });
    }
    if (!bInited_.load()) return;
    warm_up_task_();
    warm_up_finished_++;
    task_cv_.notify_all();
    {
        std::unique_lock<std::mutex> lock(warm_up_mtx_);
        task_cv_.wait(lock, [this]() {
            return warm_up_finished_ == num_clients_ || !bInited_.load();
        });
    }
    if (!bInited_.load()) return;
    SPDLOG_INFO("Warm-up finished, start to process main task. index = {}",
                client_index);
    main_task_(client_index);
}

std::unordered_map<std::string, std::string> Benchmark::get_output() {
    std::vector<std::vector<std::shared_ptr<ProfileState>>> result(
        num_clients_);

    while (!outputs_.wait_for(
        [this](size_t queue_size) { return queue_size >= total_number_; },
        SHUTDOWN_TIMEOUT_MS)) {
    }
    while (!outputs_.empty()) {
        auto item = outputs_.get();

        result[item->client_index].push_back(item);
    }

    for (auto& item : result) {
        HAMI_ASSERT(!item.empty());
        std::sort(item.begin(), item.end(), [](const auto& a, const auto& b) {
            return a->start_time < b->start_time;
        });
    }

    std::chrono::steady_clock::time_point earliest_start =
        std::chrono::steady_clock::time_point::max();
    std::chrono::steady_clock::time_point latest_end =
        std::chrono::steady_clock::time_point::min();

    std::vector<size_t> latencies;
    latencies.reserve(total_number_);
    size_t num_exception = 0;
    std::string first_exception_message = "";
    for (const auto& item : result) {
        earliest_start = std::min(earliest_start, item.front()->start_time);

        for (const auto& entry : item) {
            if (entry->exception) {
                num_exception++;
                if (first_exception_message.empty()) {
                    try {
                        std::rethrow_exception(entry->exception);
                    } catch (const std::exception& e) {
                        first_exception_message = e.what();
                    }
                }
            } else {
                latest_end = std::max(latest_end, entry->end_time);

                std::chrono::duration<float, std::milli> fp_ms =
                    entry->end_time - entry->start_time;
                latencies.emplace_back(fp_ms.count());
            }
        }
    }
    std::sort(latencies.begin(), latencies.end());
    std::unordered_map<std::string, std::string> profile_result;
    auto diff_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        latest_end - earliest_start);
    profile_result["total_time"] = std::to_string(diff_time.count());
    profile_result["num_clients"] = std::to_string(num_clients_);
    profile_result["request_batch"] = std::to_string(request_batch_);
    auto qps = size_t(latencies.size() / diff_time.count());
    profile_result["throughput::qps"] = std::to_string(qps);
    if (latencies.empty()) latencies.push_back(0);
    auto tp_50 = latencies[latencies.size() / 2];
    auto tp_99 = latencies[latencies.size() * 99 / 100];
    auto tp_90 = latencies[latencies.size() * 90 / 100];
    auto avg = std::accumulate(latencies.begin(), latencies.end(), 0) /
               (latencies.size());
    profile_result["latency::TP50"] = std::to_string(tp_50);
    profile_result["latency::TP90"] = std::to_string(tp_90);
    profile_result["latency::TP99"] = std::to_string(tp_99);
    profile_result["latency::avg"] = std::to_string(avg);
    profile_result["num_exception"] = std::to_string(num_exception);
    profile_result["first_exception_message"] = first_exception_message;
    return profile_result;
}

HAMI_REGISTER_BACKEND(Benchmark);
}  // namespace hami