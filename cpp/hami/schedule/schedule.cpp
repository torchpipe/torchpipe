#include "hami/schedule/schedule.hpp"

#include <charconv>
#include "hami/builtin/aspect.hpp"
#include "hami/builtin/proxy.hpp"
#include "hami/core/event.hpp"
#include "hami/core/helper.hpp"
#include "hami/core/parser.hpp"
#include "hami/core/queue.hpp"
#include "hami/core/reflect.h"
#include "hami/core/task_keys.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/macro.h"
#include "hami/helper/string.hpp"
#include "hami/helper/timer.hpp"

namespace hami {
namespace {
template <typename T>
std::unordered_map<T, std::string> pair2map(
    const std::vector<std::pair<T, std::string>>& config) {
  std::unordered_map<T, std::string> result;
  for (const auto& pair : config) {
    result[pair.first] = pair.second;
  }
  return result;
}
} // namespace
void Loop::impl_init(
    const std::unordered_map<string, string>& params,
    const dict& options) {
  auto [args, kwargs] = parser_v2::get_args_kwargs(this, "Loop", params);
  // str::try_update(config, "batching_timeout", batching_timeout_);
  str::try_update(kwargs, "node_name", node_name_);
  str::try_update(kwargs, "max", max_);
  str::try_update(kwargs, "timeout", timeout_);
  SPDLOG_INFO(
      "loop, node_name = {}, max = {}, timeout = {}",
      node_name_,
      max_,
      timeout_);
  HAMI_ASSERT(args.size() >= 1);
  src_queue_ = &(default_queue(args[0]));
  std::string name_dep = str::get<std::string>(kwargs, "target");
  Backend* dependency_ = HAMI_INSTANCE_GET(Backend, name_dep);
  HAMI_ASSERT(dependency_);
  HAMI_FATAL_ASSERT(dependency_->max() == std::numeric_limits<size_t>::max());
  inject_dependency(dependency_);

  bInited_.store(true);
  thread_ = std::thread(&Loop::run, this);
}

void Loop::impl_forward_sync(const std::vector<dict>& ios) {
  for (const auto& item : ios) {
    item->erase(TASK_RESULT_KEY);
  }
  // SPDLOG_INFO("impl_forward_sync start");
  injected_dependency_->forward(ios);
  // SPDLOG_INFO("impl_forward_sync end");
  return;
  std::vector<std::shared_ptr<Event>> events;
  for (const auto& item : ios) {
    auto iter = item->find(TASK_EVENT_KEY);
    events.push_back(any_cast<std::shared_ptr<Event>>(iter->second));
    item->erase(iter);
    item->erase(TASK_RESULT_KEY);
  }
  try {
    injected_dependency_->forward(ios);
  } catch (...) {
    for (std::size_t i = 0; i < ios.size(); ++i) {
      (*ios[i])[TASK_EVENT_KEY] = events[i];
      ios[i]->erase(TASK_RESULT_KEY);
    }
    for (const auto& ev : events) {
      ev->set_exception_and_notify_all(std::current_exception());
    }
  }
  for (std::size_t i = 0; i < ios.size(); ++i) {
    (*ios[i])[TASK_EVENT_KEY] = events[i];
  }
  for (const auto& ev : events) {
    ev->notify_all();
  }
}

void Loop::run() {
  std::vector<dict> input_data;
  size_t input_data_size = 0;
  bool timeout = (timeout_ == 0);

  while (bInited_.load()) {
    const auto queue_size = src_queue_->size();
    // SPDLOG_INFO(
    //     "input_data_size={}, queue_size={}", input_data_size, queue_size);
    if (input_data_size != 0 && queue_size != 0)
      SPDLOG_INFO(
          "loop, input_data_size = {}, queue_size = {}",
          input_data_size,
          queue_size);
    if (input_data_size + queue_size >= max_) {
      std::size_t new_pop = 0;
      while (input_data_size + new_pop < max_) {
        new_pop += src_queue_->front_size();
        if (input_data_size + new_pop > max_) {
          break;
        }
        input_data_size += src_queue_->front_size();
        auto data = src_queue_->get();
        input_data.push_back(data);
      }
      impl_forward_sync(input_data);
    } else if (input_data_size + queue_size == 0) {
      auto [data, size] =
          src_queue_->try_get(std::chrono::milliseconds(SHUTDOWN_TIMEOUT));
      if (data) {
        input_data.push_back(*data);
        input_data_size += size;
      }
      continue;
    } else if (timeout) {
      std::size_t new_pop = 0;
      while (!src_queue_->empty() && (input_data_size + new_pop < max_)) {
        new_pop += src_queue_->front_size();
        if (input_data_size + new_pop > max_) {
          break;
        }
        input_data_size += src_queue_->front_size();
        input_data.push_back(src_queue_->get());
      }
      if (input_data_size + new_pop >= max_) {
        continue; // go to another branch
      }
      impl_forward_sync(input_data);
    } else {
      if (input_data_size == 0) {
        input_data_size += src_queue_->front_size();
        input_data.push_back(src_queue_->get());
      }
      std::shared_ptr<Event> event =
          any_cast<std::shared_ptr<Event>>(input_data[0]->at(TASK_EVENT_KEY));
      auto time_es = event->time_passed();
      int time = int(timeout_ - time_es);
      if (time > 0) {
        if (!src_queue_->wait_for(std::chrono::milliseconds(time))) {
          timeout = true;
        } else {
          // todo
        }
      } else {
        timeout = true;
      }
      continue;
    }
    // re-init
    input_data.clear();
    input_data_size = 0;
    timeout = (timeout_ == 0);
  }
  SPDLOG_INFO("Loop exit.");
}

void Loop::impl_forward(const std::vector<dict>& ios) {
  HasEventHelper helper(
      ios); // add `event` (and wait for possible exception) if not exist

  // SPDLOG_INFO("src_queue_ puts ios.size() = {}", ios.size());
  src_queue_->puts(ios);
  // SPDLOG_INFO("src_queue_ puts finish ");
  helper.wait();
  // SPDLOG_INFO("src_queue_ wait finish ");
}

HAMI_REGISTER_BACKEND(Loop);

void Batching::impl_init(
    const std::unordered_map<string, string>& config,
    const dict& kwargs) {
  str::try_update(config, "batching_timeout", batching_timeout_);
  str::try_update(config, "node_name", node_name_);

  HAMI_ASSERT((batching_timeout_ >= 0 || batching_timeout_ == -1));

  HAMI_ASSERT(kwargs);
  instances_state_ = dict_get<std::shared_ptr<InstancesState>>(
      kwargs, TASK_RESOURCE_STATE_KEY);
  HAMI_ASSERT(instances_state_);
}

void Batching::impl_inject_dependency(Backend* dependency) {
  Dependency::impl_inject_dependency(dependency);
  const size_t max_bs = dependency->max();
  if (batching_timeout_ < 0) {
    if (max_bs > 1)
      batching_timeout_ = 4;
    else {
      batching_timeout_ = 0;
    }
  }
  if (batching_timeout_ > 0 && max_bs > 1) {
    bInited_.store(true);
    thread_ = std::thread(&Batching::run, this, max_bs);
    SPDLOG_INFO(
        "Batching impl_inject_dependency. node_name = `{}`  Max Batch Size = `{}`",
        node_name_,
        max_bs);
  } else {
    SPDLOG_INFO(
        "Batching thread not inited because batching_timeout_ = 0 or "
        "max_bs = 1");
  }
}

void Batching::impl_forward_with_dep(
    const std::vector<dict>& ios,
    Backend& dep) {
  HasEventHelper helper(
      ios); // add `event` (and wait for possible exception) if not exist

  if (bInited_.load())
    input_queue_.push(ios, get_request_size<dict>);
  else {
    dep.forward(ios);
  }
  helper.wait();
}

void Batching::run(size_t max_bs) {
  while (bInited_.load() && !input_queue_.wait_for(batching_timeout_)) {
  };
  // const size_t max_bs = max();
  SPDLOG_INFO(
      "Batching thread inited. node_name = `{}` Timeout = `{}` Max Batch Size = `{}`",
      node_name_,
      batching_timeout_,
      max_bs);
  std::vector<dict> cached_data;
  bool already_batching_timout = false;
  while (bInited_.load()) {
    const auto cached_size = get_request_size(cached_data);
    // size_t req_size =
    if (cached_size == 0) {
      dict tmp_dict;
      if (!input_queue_.wait_pop(
              tmp_dict,
              SHUTDOWN_TIMEOUT)) { // every batching_timeout_ ms check
                                   // that whether bIbited_ is true.
        // if not, exit this  loop
        continue;
      }
      cached_data.push_back(tmp_dict);
      continue;
    } else if (
        input_queue_.size() + cached_size >= max_bs ||
        already_batching_timout) {
      std::size_t new_pop = 0;
      while (cached_size + new_pop < max_bs && !input_queue_.empty()) {
        const auto front_size = input_queue_.front_size();
        if (cached_size + new_pop > max_bs) {
          break;
        }
        new_pop += front_size;
        cached_data.push_back(input_queue_.pop());
      }

      SPDLOG_DEBUG(
          "scheduler: new pop: {}, cached: {} timestamp = {}",
          new_pop,
          cached_size,
          helper::timestamp());
      if (!try_forward(cached_data, new_pop + cached_size, 1)) {
        instances_state_->wait_for(new_pop + cached_size, SHUTDOWN_TIMEOUT);
        continue;
      } else {
        already_batching_timout = false;
        cached_data.clear();
      }
    } else {
      // std::size_t new_pop = 0;

      std::shared_ptr<Event> event =
          any_cast<std::shared_ptr<Event>>(cached_data[0]->at(TASK_EVENT_KEY));
      auto time_es = int(event->time_passed());

      if (time_es < batching_timeout_) {
        if (!input_queue_.wait_for(int(batching_timeout_ - time_es))) {
          already_batching_timout = true;
        }
      } else {
        already_batching_timout = true;
      }
    }

  } // end while
}

void InstanceDispatcher::impl_init(
    const std::unordered_map<string, string>& config,
    const dict& kwargs) {
  instances_state_ = dict_get<std::shared_ptr<InstancesState>>(
      kwargs, TASK_RESOURCE_STATE_KEY);
  auto iter = config.find("node_name");
  HAMI_ASSERT(iter != config.end(), "node_name not found");
  std::string node_name = iter->second;

  size_t instance_num{1};
  str::try_update(config, "instance_num", instance_num);

  for (size_t i = 0; i < instance_num; ++i) {
    base_dependencies_.push_back(
        HAMI_INSTANCE_GET(Backend, node_name + "." + std::to_string(i)));
    HAMI_ASSERT(base_dependencies_.back());
    instances_state_->add_and_set_range(
        i, base_dependencies_.back()->min(), base_dependencies_.back()->max());
  }

  update_min_max(base_dependencies_);
}

void InstanceDispatcher::update_min_max(const std::vector<Backend*>& deps) {
  // union
  HAMI_ASSERT(max_ == 1 && min_ == std::numeric_limits<size_t>::max());
  for (const Backend* depend : deps) {
    min_ = std::min(min_, depend->min());
    max_ = std::max(max_, depend->max());
  }

  HAMI_ASSERT(min_ <= max_);
}

void InstanceDispatcher::impl_forward(const std::vector<dict>& ios) {
  HAMI_ASSERT(helper::none_or_all_has_key_and_unempty(ios, TASK_EVENT_KEY), std::to_string(ios.size()));
  const size_t req_size = get_request_size(ios);
  //
  std::string node_name = dict_get<std::string>(ios[0],"node_name", true);

  std::optional<size_t> index;
  do {
    index = instances_state_->query_available(req_size, 100, true, node_name);
  } while (!index);

  size_t valid_index{*index};
  // SPDLOG_INFO(
  //     "InstanceDispatcher, num deps = {}, req_size = {}, ios = {}",
  //     base_dependencies_.size(),
  //     req_size,
  //     ios.size());

  HAMI_FATAL_ASSERT(valid_index < base_dependencies_.size());

  std::shared_ptr<Event> event;
  auto iter = ios.back()->find(TASK_EVENT_KEY);
  if (iter != ios.back()->end()) {
    event = any_cast<std::shared_ptr<Event>>(iter->second);
  }
  if (event) {
    event->append_callback(
        [this, valid_index]() { instances_state_->remove_lock(valid_index); });
    base_dependencies_[valid_index]->forward(ios);
  } else {
    auto resource_guard = [this, valid_index](void*) {
      instances_state_->remove_lock(valid_index);
    };
    std::unique_ptr<void, decltype(resource_guard)> guard(
        nullptr, resource_guard);

    base_dependencies_[valid_index]->forward(ios);
  }
}

void BackgroundThread::impl_init(
    const std::unordered_map<string, string>& config,
    const dict& kwargs) {
  const auto dependency_name = get_dependency_name_force(this, config);

  dependency_ = std::unique_ptr<Backend>(HAMI_CREATE(Backend, dependency_name));
  HAMI_ASSERT(dependency_);

  init_task_ = [this, config, kwargs]() {
    dependency_->init(config, kwargs);
    HAMI_ASSERT(
        dependency_->min() >= 1 && dependency_->min() <= dependency_->max(),
        std::to_string(dependency_->min()) + " " +
            std::to_string(dependency_->max()));

    bInited_.store(true);
  };

  thread_ = std::thread(&BackgroundThread::run, this);
  while (!bInited_.load() && (!bStoped_.load())) {
    std::this_thread::yield();
  }
  if (init_eptr_)
    std::rethrow_exception(init_eptr_);
  HAMI_ASSERT(bInited_.load() && (!bStoped_.load()));
  SPDLOG_INFO(
      "BackgroundThread inited: {}[{}, {}]",
      dependency_name,
      dependency_->min(),
      dependency_->max());
}

void BackgroundThread::impl_forward(const std::vector<dict>& ios) {
  if (helper::all_has_key(ios, TASK_EVENT_KEY)) {
    

    batched_queue_.push(ios);
    if (ios.size() >= 1) {
      float time = helper::timestamp();
      SPDLOG_DEBUG(
          "BackgroundThread  timer: {} {} {}",
          ios.size(),
          time,
          batched_queue_.size());
    }
    std::this_thread::yield();
    return;
  }
  HAMI_ASSERT(helper::none_has_key(ios, TASK_EVENT_KEY));
  dependency_->forward(ios);
}

// void BackgroundThread::forward_task(const std::vector<dict>& ios) {}
void BackgroundThread::run() {
#ifndef NCATCH_SUB
  try {
#endif
    init_task_();

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
      auto succ = batched_queue_.wait_pop(
          tasks, SHUTDOWN_TIMEOUT); // for exit this thread

      if (!succ) {
        assert(tasks.empty());
        continue;
      }

      float time = helper::timestamp();
      SPDLOG_DEBUG(
          "batched_queue_  timer: {} {} {}",
          tasks.size(),
          time,
          batched_queue_.size());
        }

    std::vector<std::shared_ptr<Event>> events;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      auto iter_time = tasks[i]->find(TASK_EVENT_KEY);
      if (iter_time != tasks[i]->end()) {
        std::shared_ptr<Event> ti_p =
            any_cast<std::shared_ptr<Event>>(iter_time->second);
        events.push_back(ti_p);
      }
    }

    HAMI_FATAL_ASSERT(
        events.size() == tasks.size(),
        "event: " + std::to_string(events.size()) +
            " tasks: " + std::to_string(tasks.size()));
#ifndef NCATCH_SUB
    try {
#endif

      for (const auto& item : tasks) {
        item->erase(TASK_EVENT_KEY);
        item->erase(TASK_RESULT_KEY);
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

HAMI_REGISTER(Backend, BackgroundThread, "BackgroundThread");

HAMI_REGISTER(Backend, InstanceDispatcher);
HAMI_REGISTER(Backend, Batching);

class SharedInstancesState : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final {
    HAMI_ASSERT(kwargs, "kwargs is empty");
    auto res = std::make_shared<InstancesState>();
    (*kwargs)[TASK_RESOURCE_STATE_KEY] = res;
  }
};

HAMI_REGISTER_BACKEND(SharedInstancesState);

void FakeInstance::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& options) {
  int instance_num = 1;
  str::try_update(params, "instance_num", instance_num);
  HAMI_ASSERT(instance_num == 1);

  fake_instance_num_ = str::get<size_t>(params, "fake_instance_num");

  auto dep_name = parser_v2::get_dependency_name(this, params);
  parser_v2::Parser parser;
  auto deps = parser.split_by_delimiter(dep_name, ',');
  auto fake_instance_num_str = std::to_string(fake_instance_num_);

  HAMI_ASSERT(deps.size() == 1 || deps.size() == fake_instance_num_);
  dict new_options = options
      ? options
      : std::make_shared<std::unordered_map<std::string, any>>();

  for (size_t i = 0; i < fake_instance_num_; ++i) {
    auto new_params = params;
    new_params["instance_num"] = fake_instance_num_str;
    new_params[TASK_INDEX_KEY] = std::to_string(i);
    auto backend = init_backend(
        deps.size() == 1 ? deps[0] : deps[i], new_params, new_options);
    backends_.push_back(std::move(backend));
  }

  std::vector<size_t> mins;
  std::vector<size_t> maxs;
  for (std::size_t i = 0; i < backends_.size(); ++i) {
    mins.push_back(backends_[i]->min());
    maxs.push_back(backends_[i]->max());
  }
  min_ = *std::min_element(mins.begin(), mins.end());
  max_ = *std::max_element(maxs.begin(), maxs.end());

  HAMI_ASSERT(min() <= max(), " min() = {} max() = {}", min(), max());

  sorted_max_ = sort_indexes(backends_);
}

void FakeInstance::impl_forward(const std::vector<dict>& ios) {
  const auto size = get_request_size(ios);

  const auto index = get_best_match(size);
  // SPDLOG_INFO("FakeInstance: size={} index={}", size, index);
  // auto ios = ios_data;
  if (size != ios.size()) {
    IPIPE_ASSERT(index >= 0); // garentied
    backends_[index]->forward(ios);
  } else {
    if (index < 0) {
      backends_[0]->forward(ios);
    } // todo: split?
    else
      backends_[index]->forward(ios);
  }
}
HAMI_REGISTER_BACKEND(FakeInstance);

} // namespace hami
