#include <charconv>
#include "hami/core/event.hpp"
#include "hami/helper/base_logging.hpp"

#include "hami/schedule/schedule.hpp"
#include "hami/builtin/aspect.hpp"
#include "hami/builtin/proxy.hpp"
#include "hami/helper/string.hpp"
#include "hami/helper/macro.h"
#include "hami/core/helper.hpp"
#include "hami/core/task_keys.hpp"
#include "hami/helper/string.hpp"
namespace hami {
void BatchingEvent::pre_init(const std::unordered_map<string, string>& config,
                             const dict& dict_config) {
  if (config.find("batching_timeout") != config.end()) {
    batching_timeout_ = stoi(config.at("batching_timeout"));
  } else {
    batching_timeout_ = 2;
    SPDLOG_WARN("batching_timeout not found, using default value: {}", batching_timeout_);
  }
  HAMI_ASSERT(batching_timeout_ >= 0);
  if (batching_timeout_ > 0) {
    bInited_.store(true);
    thread_ = std::thread(&BatchingEvent::run, this);
    SPDLOG_INFO("BatchingEvent thread inited. Timeout = {}, Max Batch Size = {}", batching_timeout_,
                max());
  } else {
    SPDLOG_INFO("BatchingEvent thread not inited. Timeout = {}, Max Batch Size = {}",
                batching_timeout_, max());
  }
}

void BatchingEvent::forward_impl(const std::vector<dict>& inputs, Backend* dependency) {
  HAMI_ASSERT(injected_dependency_ == dependency);
  if (bInited_.load())
    input_queue_.push(inputs, get_request_size<dict>);
  else {
    injected_dependency_->forward(inputs);
  }
}

void BatchingEvent::run() {
  const size_t max_bs = max();
  std::vector<dict> cached_data;

  while (bInited_.load()) {
    const auto cached_size = get_request_size(cached_data);

    if (input_queue_.size() + cached_size >= max_bs) {
      std::size_t new_pop = 0;
      while (cached_size + new_pop < max_bs) {
        new_pop += input_queue_.front_size();
        if (cached_size + new_pop > max_bs) {
          break;
        }
        cached_data.push_back(input_queue_.pop());
      }

      if (!injected_dependency_->try_forward(cached_data, SHUTDOWN_TIMEOUT)) continue;
    } else if (cached_size + input_queue_.size() == 0) {
      dict tmp_dict;
      if (!input_queue_.wait_pop(tmp_dict,
                                 SHUTDOWN_TIMEOUT)) {  // every batching_timeout_ ms check that
                                                       // whether bIbited_ is true.
        // if not, exit this  loop
        continue;
      }
      cached_data.push_back(tmp_dict);
      continue;

    } else {
      std::size_t new_pop = 0;

      if (cached_size == 0) {
        HAMI_FATAL_ASSERT(!input_queue_.empty());
        new_pop += input_queue_.front_size();
        cached_data.push_back(input_queue_.pop());
        // it is guaranteed that new_pop < max_bs for one input
      }
      std::shared_ptr<Event> event =
          any_cast<std::shared_ptr<Event>>(cached_data[0]->at(TASK_EVENT_KEY));
      auto time_es = event->time_passed();

      if (time_es < batching_timeout_) {
        input_queue_.wait(int(batching_timeout_ - time_es));
        continue;
      }

      while (!input_queue_.empty() && (cached_size + new_pop < max_bs)) {
        new_pop += input_queue_.front_size();
        if (cached_size + new_pop > max_bs) {
          // new_pop -= input_queue_.front_size();
          break;
        }
        cached_data.push_back(input_queue_.pop());
      }
      if (cached_size + new_pop > max_bs) {
        continue;  // go to another branch
      }

      if (!injected_dependency_->try_forward(cached_data, SHUTDOWN_TIMEOUT)) continue;
    }
    cached_data.clear();
  }  // end while
}

HAMI_PROXY(Batching, "Aspect[EventGuard,BatchingEvent]");

void InstanceDispatcher::init(const std::unordered_map<string, string>& config,
                              const dict& dict_config) {
  instances_state_ = std::make_unique<InstancesState>();
  auto iter = config.find("node_name");
  HAMI_ASSERT(iter != config.end(), "node_name not found");
  std::string node_name = iter->second;

  size_t instance_num{1};
  str::str2int(config, "instance_num", instance_num);

  for (size_t i = 0; i < instance_num; ++i) {
    base_dependencies_.push_back(HAMI_INSTANCE_GET(Backend, node_name + "." + std::to_string(i)));
    HAMI_ASSERT(base_dependencies_.back());
    instances_state_->add(i, base_dependencies_.back()->min(), base_dependencies_.back()->max());
  }

  // std::sort(base_dependencies_.begin(), base_dependencies_.end(),
  //           [](const Backend* a, const Backend* b) { return a->max() >= b->max(); });

  auto [min_, max_] = update_min_max(base_dependencies_);
}

std::pair<size_t, size_t> InstanceDispatcher::update_min_max(const std::vector<Backend*>& depends) {
  // union
  size_t max_value = 1;
  size_t min_value = std::numeric_limits<size_t>::max();

  for (const Backend* depend : depends) {
    min_value = std::min(min_value, depend->min());
    max_value = std::max(max_value, depend->max());
  }

  HAMI_ASSERT(min_value <= max_value);
  return {min_value, max_value};
}

bool InstanceDispatcher::try_forward(const std::vector<dict>& inputs, size_t timeout) {
  const size_t req_size = get_request_size(inputs);
  size_t valid_index{};
  if (!instances_state_->wait_resource(req_size, valid_index, timeout)) {
    SPDLOG_INFO("InstanceDispatcher::try_forward, req_size={}", req_size);

    return false;
  }
  SPDLOG_INFO("InstanceDispatcher::try_forward success, req_size={}", req_size);

  HAMI_ASSERT(valid_index < base_dependencies_.size(),
              "InstanceDispatcher: no valid backend found. req_size={}", req_size);

  std::shared_ptr<Event> event;
  auto iter = inputs.back()->find(TASK_EVENT_KEY);
  if (iter != inputs.back()->end()) {
    event = any_cast<std::shared_ptr<Event>>(iter->second);
  }
  if (event) {
    event->set_callback(
        [this, valid_index]() { instances_state_->finished_instance(valid_index); });
    base_dependencies_[valid_index]->forward(inputs);
  } else {
    auto resource_guard = [this, valid_index](void*) {
      instances_state_->finished_instance(valid_index);
    };
    std::unique_ptr<void, decltype(resource_guard)> guard(nullptr, resource_guard);

    base_dependencies_[valid_index]->forward(inputs);
  }
  return true;
}

void BackgroundThreadEvent::init(const std::unordered_map<string, string>& config,
                                 const dict& dict_config) {
  constexpr auto default_name = "BackgroundThreadEvent";
  auto name = HAMI_OBJECT_NAME(Backend, this);
  if (name == std::nullopt) {
    name = default_name;
    SPDLOG_WARN(
        "{}::init, it seems this instance was not created via reflection, using default name {}. "
        "Please configure its dependency via the parameter {}::dependency",
        *name, *name, *name);
  }
  auto iter = config.find(*name + "::dependency");
  if (iter != config.end()) {
    dependency_name_ = iter->second;
    config_ = &config;
    dict_config_ = dict_config;
  } else {
    SPDLOG_WARN(
        "Dependency configuration {}::dependency not found, skipping dependency injection process",
        *name);
    throw std::runtime_error("BackgroundThreadEvent: no dependency found");
  }

  thread_ = std::thread(&BackgroundThreadEvent::run, this);
  while (!bInited_.load() && (!bStoped_.load())) {
    std::this_thread::yield();
  }
  if (init_eptr_) std::rethrow_exception(init_eptr_);
  HAMI_ASSERT(bInited_.load() && (!bStoped_.load()));
}

void BackgroundThreadEvent::run() {
#ifndef NCATCH_SUB
  try {
#endif

    dependency_ = std::unique_ptr<Backend>(HAMI_CREATE(Backend, dependency_name_));
    HAMI_ASSERT(dependency_, "`" + dependency_name_ + "` is not a valid backend");
    dependency_->init(*config_, dict_config_);

    bInited_.store(true);

#ifndef NCATCH_SUB
  } catch (const std::exception& e) {
    bInited_.store(false);
    dependency_.reset();
    SPDLOG_ERROR("Backend initialization: {}", e.what());
    init_eptr_ = std::current_exception();
  }
#endif

  while (bInited_.load()) {
    std::vector<dict> tasks;
    {
      auto succ = batched_queue_.wait_pop(tasks, 50);  // for exit this thread

      if (!succ) {
        assert(tasks.empty());
        continue;
      }
    }
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      tasks[i]->erase(TASK_RESULT_KEY);
    }

    std::vector<std::shared_ptr<Event>> events;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      auto iter_time = tasks[i]->find(TASK_EVENT_KEY);
      if (iter_time != tasks[i]->end()) {
        std::shared_ptr<Event> ti_p = any_cast<std::shared_ptr<Event>>(iter_time->second);
        events.push_back(ti_p);
      }
    }
    HAMI_FATAL_ASSERT(events.size() == tasks.size());
#ifndef NCATCH_SUB
    try {
#endif
      HAMI_FATAL_ASSERT(tasks.size() >= min() || tasks.size() <= max());
      for (auto item : tasks) {
        item->erase(TASK_EVENT_KEY);
      }
      dependency_->forward(tasks);

#ifndef NCATCH_SUB
    } catch (...) {
      for (std::size_t i = 0; i < tasks.size(); ++i) {
        (*tasks[i])[TASK_EVENT_KEY] = events[i];
      }
      for (std::size_t i = 0; i < tasks.size(); ++i) {
        events[i]->set_exception_and_notify_all(std::current_exception());
      }
      continue;
    }
#endif
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      (*tasks[i])[TASK_EVENT_KEY] = events[i];
    }
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      events[i]->notify_all();
    }
  }
  bStoped_.store(true);
}

HAMI_REGISTER(Backend, BackgroundThreadEvent, "BackgroundThreadEvent");
HAMI_PROXY(BackgroundThread, "Aspect[EventGuard,BackgroundThreadEvent]");

HAMI_REGISTER(Backend, InstanceDispatcher);
HAMI_REGISTER(Backend, BatchingEvent);
}  // namespace hami