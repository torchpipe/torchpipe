#include "hami/schedule/schedule.hpp"

#include <charconv>

#include "hami/builtin/aspect.hpp"
#include "hami/builtin/proxy.hpp"
#include "hami/core/event.hpp"
#include "hami/core/helper.hpp"
#include "hami/core/queue.hpp"
#include "hami/core/reflect.h"
#include "hami/core/task_keys.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/macro.h"
#include "hami/helper/string.hpp"
#include "hami/helper/timer.hpp"

namespace hami {

void Loop::impl_init(
    const std::unordered_map<string, string>& params,
    const dict& options) {
  auto [args, kwargs] = parser_v2::get_args_kwargs(this, "Loop", params);
  // str::try_update(config, "batching_timeout", batching_timeout_);
  str::try_update(kwargs, "node_name", node_name_);
  HAMI_ASSERT(args.size() >= 1);
  src_queue_ = &(default_queue(args[0]));
  std::string name_dep = str::update<std::string>(kwargs, "target");
  Backend* dependency_ = HAMI_INSTANCE_GET(Backend, name_dep);
  HAMI_ASSERT(dependency_);
  HAMI_FATAL_ASSERT(dependency_->max() == std::numeric_limits<size_t>::max());
  inject_dependency(dependency_);

  bInited_.store(true);
  thread_ = std::thread(&Loop::run, this);
}

void Loop::run() {
  std::vector<dict> datas;

  while (bInited_.load()) {
    do {
      auto [data, size] =
          src_queue_->try_get(std::chrono::milliseconds(SHUTDOWN_TIMEOUT));

      if (data) {
        datas.push_back(*data);
      }
    } while (!src_queue_->empty());
    if (!datas.empty()) {
      // todo safe_forward
      injected_dependency_->forward(datas);
      datas.clear();
    }
  }
}

void Loop::impl_forward(const std::vector<dict>& inputs) {
  HasEventHelper helper(
      inputs); // add `event` (and wait for possible exception) if not exist

  src_queue_->puts(inputs);
  helper.wait();
}

HAMI_REGISTER_BACKEND(Loop);

void Batching::impl_init(
    const std::unordered_map<string, string>& config,
    const dict& kwargs) {
  str::try_update(config, "batching_timeout", batching_timeout_);
  str::try_update(config, "node_name", node_name_);

  HAMI_ASSERT(batching_timeout_ >= 0);

  HAMI_ASSERT(kwargs);
  instances_state_ = dict_get<std::shared_ptr<InstancesState>>(
      kwargs, TASK_RESOURCE_STATE_KEY);
  HAMI_ASSERT(instances_state_);
}

void Batching::impl_inject_dependency(Backend* dependency) {
  Dependency::impl_inject_dependency(dependency);
  const size_t max_bs = dependency->max();
  if (batching_timeout_ > 0 && max_bs > 1) {
    bInited_.store(true);
    thread_ = std::thread(&Batching::run, this);
  } else {
    SPDLOG_INFO(
        "Batching thread not inited because batching_timeout_ = 0 or "
        "max_bs = 1");
  }
}

void Batching::impl_forward_with_dep(
    const std::vector<dict>& inputs,
    Backend* dep) {
  HasEventHelper helper(
      inputs); // add `event` (and wait for possible exception) if not exist
  HAMI_FATAL_ASSERT(dep);

  if (bInited_.load())
    input_queue_.push(inputs, get_request_size<dict>);
  else {
    dep->forward(inputs);
  }
  helper.wait();
}

void Batching::run() {
  while (bInited_.load() && !input_queue_.wait_for(batching_timeout_)) {
  };
  const size_t max_bs = max();
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

      if (!try_forward(cached_data, new_pop + cached_size, SHUTDOWN_TIMEOUT))
        continue;
      else {
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

void InstanceDispatcher::impl_forward(const std::vector<dict>& inputs) {
  HAMI_ASSERT(helper::none_or_all_has_key_and_unempty(inputs, TASK_EVENT_KEY));
  const size_t req_size = get_request_size(inputs);
  //
  std::optional<size_t> index;
  do {
    index = instances_state_->query_avaliable(req_size, 100, true);
  } while (!index);
  size_t valid_index{*index};

  HAMI_FATAL_ASSERT(valid_index < base_dependencies_.size());

  std::shared_ptr<Event> event;
  auto iter = inputs.back()->find(TASK_EVENT_KEY);
  if (iter != inputs.back()->end()) {
    event = any_cast<std::shared_ptr<Event>>(iter->second);
  }
  if (event) {
    event->append_callback(
        [this, valid_index]() { instances_state_->remove_lock(valid_index); });
    base_dependencies_[valid_index]->forward(inputs);
  } else {
    auto resource_guard = [this, valid_index](void*) {
      instances_state_->remove_lock(valid_index);
    };
    std::unique_ptr<void, decltype(resource_guard)> guard(
        nullptr, resource_guard);

    base_dependencies_[valid_index]->forward(inputs);
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

void BackgroundThread::impl_forward(const std::vector<dict>& inputs) {
  HAMI_ASSERT(helper::all_has_key(inputs, TASK_EVENT_KEY));
  batched_queue_.push(inputs);
}

// void BackgroundThread::forward_task(const std::vector<dict>& inputs) {}
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

// struct CBProtocol {
//   enum struct Action { Stop, Cancel };
//   id_type req_id;
//   int32_t req_tokens;
//   int32_t new_tokens;
//   int32_t max_new_tokens;
//   int32_t max_tokens;
//   std::vector<int32_t> stop_token_ids;
//   Action action;
// };

void ContiguousBatching::impl_init(
    const std::unordered_map<string, string>& params,
    const dict& options) {
  auto [args, kwargs] =
      parser_v2::get_args_kwargs(this, "ContiguousBatching", params);
  std::string target = str::update<std::string>(kwargs, "target");
  dependency_ = HAMI_INSTANCE_GET(Backend, target);
  HAMI_ASSERT(dependency_, target + " not found (ContiguousBatching).");
}

void ContiguousBatching::impl_forward(const std::vector<dict>& io) {
  std::vector<CBProtocol> configs;
  configs.reserve(io.size());
  for (const auto& item : io) {
    CBProtocol pro;
    pro.req_id = dict_get<std::string>(item, TASK_REQUEST_ID_KEY);
    // HAMI_FATAL_ASSERT(item->find(TASK_MSG_KEY) != item->end());
    auto re = dict_get<std::shared_ptr<TypedDict>>(item, TASK_MSG_KEY);
    parser_message(re, pro);

    configs.emplace_back(std::move(pro));

    item->erase(TASK_MSG_KEY);
  }
  dependency_->forward(io);
}

void ContiguousBatching::parser_message(
    const std::shared_ptr<TypedDict>& msg,
    CBProtocol& pro) {
  pro.req_tokens = get<int32_t>(*msg, "req_tokens");
  pro.max_tokens = get<int32_t>(*msg, "max_tokens");
  pro.new_tokens = get<int32_t>(*msg, "new_tokens");
  try_update<int32_t>(*msg, "max_new_tokens", pro.max_new_tokens);
  SPDLOG_INFO(
      "\n"
      "+---------------------------- Contiguous Batching ----------------------------+\n"
      "| Request ID:      {:45} |\n"
      "| Req Tokens:      {:45} |\n"
      "| Max Tokens:      {:45} |\n"
      "| New Tokens:      {:45} |\n"
      "| Max New Tokens:  {:45} |\n"
      "+------------------------------------------------------------------------------+",
      pro.req_id,
      pro.req_tokens,
      pro.max_tokens,
      pro.new_tokens,
      pro.max_new_tokens);
}

HAMI_REGISTER_BACKEND(ContiguousBatching);
} // namespace hami