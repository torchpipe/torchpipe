#include "omniback/builtin/benchmark.hpp"
#include <numeric>
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/string.hpp"
#include "omniback/helper/timer.hpp"

#include "omniback/builtin/proxy.hpp"
#include "omniback/builtin/result_queue.hpp"
#include "omniback/builtin/source.hpp"
#include "omniback/core/event.hpp"
#include "omniback/core/helper.hpp"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/macro.h"
#include <tvm/ffi/extra/stl.h>

#include "omniback/builtin/generate_backend.hpp"
namespace omniback {

void Benchmark::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  auto dep = get_dependency_name(this, config, "Benchmark");
  std::string tag = dep? dep.value(): "";
  target_queue_ = &default_queue(tag);

  OMNI_ASSERT(target_queue_);

  str::try_update(config, "num_clients", num_clients_);
  str::try_update(config, "request_batch", request_batch_);
  total_number_ = str::get<size_t>(config, "total_number");
  str::try_update(config, "num_warm_up", num_warm_up_);

  if (total_number_ % request_batch_ != 0) {
    total_number_ += request_batch_ - total_number_ % request_batch_;
    SPDLOG_WARN(
        "total_number is not divisible by request_batch, so we change it "
        "to {}",
        total_number_);
  }

  inputs_ =
      std::make_unique<queue::ThreadSafeQueue<std::shared_ptr<ProfileState>>>();

  bInited_.store(true);

  for (size_t i = 0; i < num_clients_; i++) {
    threads_.emplace_back(&Benchmark::run, this, i);
  }

  // stages_.resize(num_clients_, Stage::WaitingForWarmup)
}

void Benchmark::impl_forward_with_dep(
    const std::vector<dict>& input,
    Backend& dependency) {
  OMNI_ASSERT(input.size() > 1);

  {
    // warm up
    std::unique_lock<std::mutex> lock(warm_up_mtx_);

    warm_up_task_ = [this, input, &dependency]() {
      int num_warm_up = num_warm_up_;

      while (num_warm_up-- > 0) {
        std::vector<dict> warm_data;
        for (size_t i = 0; i < request_batch_; ++i) {
          warm_data.push_back(uniform_sample(input));
        }
        try {
          dependency.forward(warm_data);
        } catch (const std::exception& e) {
          SPDLOG_ERROR("Exception during warm-up forward: {}", e.what());
          break;
        }
      }
    };

    main_task_ = [this, &dependency](size_t client_index) {
      while (bInited_.load()) {
        auto item = inputs_->try_get(SHUTDOWN_TIMEOUT_MS);

        if (item) {
          // SPDLOG_INFO("input = {}", inputs_->size());
          auto data = *item;
          auto& state = *(data);

          state.client_index = client_index;
          state.arrive_time = std::chrono::steady_clock::now();

          state.start_time = std::chrono::steady_clock::now();
          std::exception_ptr excep;
          try {
            dependency.forward(state.data);
          } catch (...) {
            state.exception = (std::current_exception());
          }
          state.end_time = std::chrono::steady_clock::now();
          state.data.clear();
          outputs_.put(data);
        } else if (bNoNewData_.load() && inputs_->empty()) {
          break;
        }
      }
    };
  }

  task_cv_.notify_all();

  // generate test data
  size_t req_times = total_number_ / request_batch_;
  // size_t req_times = num_warm_up_ * request_batch_;
  while (req_times-- > 0) {
    auto data = std::make_shared<ProfileState>();
    data->arrive_time = std::chrono::steady_clock::now();

    for (size_t i = 0; i < request_batch_; ++i) {
      data->data.push_back(uniform_sample(input));
    }
    while (!inputs_->try_put(
        data,
        num_clients_ + 100,
        std::chrono::milliseconds(SHUTDOWN_TIMEOUT))) {
    };
    // SPDLOG_INFO("req_times = {} inputs_ size {}", req_times,
    //             inputs_->size());
  }
  bNoNewData_.store(true);
  std::exception_ptr first_exception;
  auto profile_result = get_output(first_exception);
  dict data = make_dict();
  (*data)[TASK_DATA_KEY] = profile_result;
  target_queue_->push(data);

  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
  // get outputs
}

void Benchmark::run(size_t client_index) {
  OMNI_ASSERT(bInited_.load());
  {
    std::unique_lock<std::mutex> lock(warm_up_mtx_);
    task_cv_.wait(lock, [this]() { return warm_up_task_ || !bInited_.load(); });
  }
  if (!bInited_.load())
    return;
  warm_up_task_();
  warm_up_finished_++;
  task_cv_.notify_all();
  {
    std::unique_lock<std::mutex> lock(warm_up_mtx_);
    task_cv_.wait(lock, [this]() {
      return warm_up_finished_ == num_clients_ || !bInited_.load();
    });
  }
  if (!bInited_.load())
    return;
  SPDLOG_INFO(
      "Warm-up finished, start to process main task. index = {}", client_index);
  main_task_(client_index);
}

std::unordered_map<std::string, std::string> Benchmark::get_output(
    std::exception_ptr& first_exception) {
  std::vector<std::vector<std::shared_ptr<ProfileState>>> result(num_clients_);

  while (!outputs_.wait_for_new_data(
      [this](size_t queue_size) { return queue_size >= total_number_; },
      SHUTDOWN_TIMEOUT_MS)) {
  }
  while (!outputs_.empty()) {
    auto item = outputs_.get();

    result[item->client_index].push_back(item);
  }

  for (auto& item : result) {
    OMNI_ASSERT(!item.empty());
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

        if (!first_exception)
          first_exception = entry->exception;
        // todo

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
  auto qps = double(latencies.size() * 1000.0 / diff_time.count());
  profile_result["throughput::qps"] = std::to_string(size_t(qps));
  if (latencies.empty())
    latencies.push_back(0);
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

OMNI_REGISTER_BACKEND(Benchmark);

class Profile : public Dependency {
 public:
  struct Status {
    // size_t client_index;
    size_t thread_id;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
  };

 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override {
    // str::try_update(params, "num_clients", num_clients_);
    // str::try_update(params, "request_batch", request_batch_);
    // total_number_ = str::update<size_t>(config, "total_number");
    // str::try_update(params, "num_warm_up", num_warm_up_);

    target_queue_ = &default_queue();
  }

  void impl_forward_with_dep(const std::vector<dict>& io, Backend& dep)
      override {
    thread_local const std::string thread_id = std::to_string(
        std::hash<std::thread::id>()(std::this_thread::get_id()));

    std::vector<std::string> req_ids;
    for (const auto& item : io) {
      id_type req_id = dict_get<id_type>(item, TASK_REQUEST_ID_KEY);
      req_ids.push_back(req_id);
    }

    // size_t req_size = io.size();
    static const auto first_time = std::chrono::steady_clock::now();
    TypedDict status;
    status.data[TASK_REQUEST_ID_KEY] = req_ids;
    status.data["thread_id"] = thread_id;
    status.data["start_time"] =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - first_time)
            .count();
    dep.forward(io);
    // try {
    //   dep.forward(io);
    // } catch (const std::exception& e) {
    //   SPDLOG_WARN("Exception during forward: {}", e.what());
    //   status.data["exception"] = std::string(e.what());

    //   status.data["end_time"] =
    //       std::chrono::duration_cast<std::chrono::duration<double>>(
    //           std::chrono::steady_clock::now() - first_time)
    //           .count();
    //   auto data = make_dict();
    //   data->insert({TASK_DATA_KEY, status});
    //   target_queue_->push_wo_notify(data);
    //   throw;
    // }

    status.data["end_time"] =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - first_time)
            .count();
    auto data = make_dict();
    data->insert({TASK_DATA_KEY, status});
    target_queue_->push_wo_notify(data);
  }

 private:
  // size_t request_batch_ = 1;
  // size_t total_number_ = 10000;
  // size_t num_warm_up_ = 20;
  Queue* target_queue_{nullptr};
};
OMNI_REGISTER_BACKEND(Profile);

} // namespace omniback