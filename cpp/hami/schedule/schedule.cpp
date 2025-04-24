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

void Loop::impl_forward(const std::vector<dict>& inputs) {
  HasEventHelper helper(
      inputs); // add `event` (and wait for possible exception) if not exist

  // SPDLOG_INFO("src_queue_ puts inputs.size() = {}", inputs.size());
  src_queue_->puts(inputs);
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
    index = instances_state_->query_available(req_size, 100, true);
  } while (!index);
  size_t valid_index{*index};
  // SPDLOG_INFO(
  //     "InstanceDispatcher, num deps = {}, req_size = {}, inputs = {}",
  //     base_dependencies_.size(),
  //     req_size,
  //     inputs.size());

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

// struct BatchInfo {
//   enum struct Action { Stop, Cancel };
//   id_type req_id;
//   int req_tokens;
//   int new_tokens;
//   int max_tokens;
//   int context_length;
//   std::vector<int> stop_token_ids;
//   Action action;
// };

void ContiguousBatching::impl_init(
    const std::unordered_map<string, string>& params,
    const dict& options) {
  auto [args, kwargs] =
      parser_v2::get_args_kwargs(this, "ContiguousBatching", params);
  std::string target = str::get<std::string>(kwargs, "target");
  max_ = str::get<int>(kwargs, "max");
  // auto no_page_table = str::get(kwargs, "no_page_table");
  // no_page_table_ = init_backend(no_page_table, params, options);
  SPDLOG_INFO("contiguous batching, target = {}, max = {}", target, max_);

  dependency_ = HAMI_INSTANCE_GET(Backend, target);
  HAMI_ASSERT(dependency_, target + " not found (ContiguousBatching).");

  page_table_ = &default_page_table();
  page_size_ = page_table_->page_size();
  HAMI_ASSERT(page_size_ > 0);
}
std::pair<std::vector<id_type>, std::unordered_map<id_type, std::string>>
ContiguousBatching::get_activated_ids() {
  std::vector<std::pair<id_type, std::string>> will_finish_ids;
  std::vector<std::pair<id_type, std::string>> will_finish_ids_prefill;
  std::vector<id_type> wont_finish_ids;
  std::vector<id_type> prefill_ids;
  std::unordered_map<id_type, int> new_page_needed;
  int will_finish_ids_new_pages{0}, wont_finish_ids_new_pages{0},
      wont_finish_ids_new_pages_next_round{0}; // decode
  int available_ids = page_table_->available_ids();
  int available_pages = page_table_->available_pages();
  for (auto it = req_status_.begin(); it != req_status_.end(); ++it) {
    const auto& info = it->second;
    SPDLOG_DEBUG(
        "id={} req_tokens={},new_tokens={}, context_length={}, max_tokens={}",
        it->first,
        info.req_tokens,
        info.new_tokens,
        info.context_length,
        info.max_tokens);

    if ((info.req_tokens + info.new_tokens + 1 == info.context_length) ||
        info.new_tokens + 1 == info.max_tokens) {
      if (it->second.new_tokens == 0) {
        // prefill
        prefill_ids.push_back(it->first);
        will_finish_ids_prefill.push_back({it->first, "length"});
        new_page_needed[it->first] =
            (info.req_tokens + page_size_ - 1) / page_size_;
      } else {
        // decode
        will_finish_ids.push_back({it->first, "length"});
        new_page_needed[it->first] =
            (1 == (info.req_tokens + info.new_tokens) % page_size_);
        will_finish_ids_new_pages += new_page_needed[it->first];
      }
    } else if (it->second.new_tokens == 0) { // prefill
      prefill_ids.push_back(it->first);
      new_page_needed[it->first] =
          (info.req_tokens + page_size_ - 1) / page_size_;
    } else {
      wont_finish_ids.push_back(it->first);
      new_page_needed[it->first] =
          (1 == (info.req_tokens + info.new_tokens) % page_size_);
      wont_finish_ids_new_pages += new_page_needed[it->first];
      wont_finish_ids_new_pages_next_round +=
          (1 == (info.req_tokens + info.new_tokens + 1) % page_size_);
    }
  }
  int num_decodes = wont_finish_ids.size() + will_finish_ids.size();
  wont_finish_ids_new_pages_next_round += wont_finish_ids_new_pages;

  // limited by number of id
  stable_sort_by_time(prefill_ids);
  if (available_ids < prefill_ids.size())
    prefill_ids.resize(available_ids);

  // limited by number of page for prefill
  if (wont_finish_ids_new_pages_next_round + will_finish_ids_new_pages >=
      available_pages) {
    prefill_ids.clear();
  } else {
    int max_prefill_pages =
        (available_pages - wont_finish_ids_new_pages_next_round -
         will_finish_ids_new_pages);
    // SPDLOG_INFO(
    //     "available_pages={}, will_finish_ids_new_pages={},
    //     wont_finish_ids_new_pages_next_round={}", available_pages,
    //     will_finish_ids_new_pages,
    //     wont_finish_ids_new_pages_next_round);
    int max_prefill_size = 0;
    for (const auto& id : prefill_ids) {
      auto max_token = req_status_[id].max_tokens != 0
          ? req_status_[id].max_tokens + req_status_[id].req_tokens
          : req_status_[id].context_length;
      HAMI_FATAL_ASSERT(max_token > 0 && max_token < INT_MAX);
      const int max_pages = (max_token + page_size_ - 1) / page_size_;
      // SPDLOG_INFO(
      //     "prefill id={}, max_token={}, max_pages={}",
      //     id,
      //     max_token,
      //     max_pages);
      if (max_prefill_pages >= max_pages) {
        max_prefill_pages -= max_pages;
        max_prefill_size++;
      } else {
        break;
      }
    }
    if (max_prefill_size < prefill_ids.size()) {
      prefill_ids.resize(max_prefill_size);
    }
  }
  // limited by `max`(max batch size of the network) for prefill (prefill first
  // for batch size limitation)
  size_t valid_count = 0;
  int max_prefill_batch_size = 0;
  for (const auto& id : prefill_ids) {
    if (max_prefill_batch_size + req_status_[id].req_tokens > max_) {
      break; // 停止累加后续 ID
    }
    max_prefill_batch_size += req_status_[id].req_tokens;
    valid_count++;
  }
  prefill_ids.resize(valid_count);

  // limited by `max`(max batch size of the network) for decode
  int max_decode_batch_size = max_ - max_prefill_batch_size;
  if (max_decode_batch_size < will_finish_ids.size()) {
    will_finish_ids.resize(max_decode_batch_size);
    wont_finish_ids.clear();
  } else if (
      max_decode_batch_size < will_finish_ids.size() + wont_finish_ids.size()) {
    wont_finish_ids.resize(max_decode_batch_size - will_finish_ids.size());
  }

  // limited by number of page for decode
  std::vector<id_type> dropped_decode_ids;
  if (will_finish_ids_new_pages + wont_finish_ids_new_pages > available_pages) {
    HAMI_ASSERT(prefill_ids.empty());
    {
      // parse will_finish_ids
      stable_sort_by_time(will_finish_ids);
      for (auto iter = will_finish_ids.begin();
           iter != will_finish_ids.end();) {
        if (new_page_needed[iter->first] == 0) {
          iter++;
          continue;
        }
        HAMI_FATAL_ASSERT(
            new_page_needed[iter->first] == 1,
            iter->first + ": " +
                std::to_string(new_page_needed.at(iter->first)));
        if (available_pages == 0) {
          dropped_decode_ids.push_back(iter->first);
          iter = will_finish_ids.erase(iter);

        } else {
          available_pages--;
          iter++;
        }
      }
    }
    {
      // parse wont_finish_ids
      stable_sort_by_time(wont_finish_ids);
      for (auto iter = wont_finish_ids.begin();
           iter != wont_finish_ids.end();) {
        if (new_page_needed[*iter] == 0) {
          iter++;
          continue;
        }
        HAMI_FATAL_ASSERT(new_page_needed[*iter] == 1);
        if (available_pages == 0) {
          dropped_decode_ids.push_back(*iter);
          iter = wont_finish_ids.erase(iter);
        } else {
          available_pages--;
          iter++;
        }
      }
    }
  }
  if (!prefill_ids.empty() || !dropped_decode_ids.empty() ||
      !will_finish_ids.empty())
    SPDLOG_INFO(
        "[Drop: decode_ids={}] [activated: will_finish_ids={}] [prefill_ids={}]",
        str::join(dropped_decode_ids),
        str::join(will_finish_ids),
        str::join(prefill_ids));

  // Controls Droping mechanism for unfinished inference tasks (wont_finish_ids)
  // when insufficient memory pages exist for future processing.

  /* Drop Strategy Options:
   * -------------------------
   * [Trigger Condition] When should we drop (offload to CPU or trigger
   * re-computation)?
   *
   * Option A: drop one when available_pages == 0 for the next round
   *
   * Option B: drop one when dropped decode ids(because of no pages) > 0.4 of
   * total decodes
   */
  if (will_finish_ids.empty() && prefill_ids.empty()) {
    HAMI_FATAL_ASSERT(
        !wont_finish_ids.empty(),
        "wont_finish_ids_new_pages_next_round =" +
            std::to_string(wont_finish_ids_new_pages_next_round) +
            " available_pages=" + std::to_string(available_pages));
    const float alpha = 0.4;
    if (available_pages <= 0) {
      const auto& info = req_status_.at(wont_finish_ids.back());
      if ((info.new_tokens + info.req_tokens) % page_size_ == 0) {
        will_finish_ids.push_back({wont_finish_ids.back(), "no_page"});
        wont_finish_ids.pop_back();
      }
    } else if (dropped_decode_ids.size() >= num_decodes * alpha) {
      will_finish_ids.push_back({wont_finish_ids.back(), "no_page"});
      wont_finish_ids.pop_back();
    }
  }

  // combine final result
  prefill_ids.reserve(
      prefill_ids.size() + wont_finish_ids.size() + will_finish_ids.size());
  prefill_ids.insert(
      prefill_ids.end(), wont_finish_ids.begin(), wont_finish_ids.end());
  for (const auto& id : will_finish_ids) {
    prefill_ids.push_back(id.first);
  }
  auto will_finish_map = pair2map(will_finish_ids);
  for (const auto& item : will_finish_ids_prefill) {
    will_finish_map[item.first] = item.second;
  }
  return {prefill_ids, will_finish_map};
}

void ContiguousBatching::impl_forward(const std::vector<dict>& io) {
  // only one thread can call this funciton
  // process msg
  for (const auto& item : io) {
    auto req_id = dict_get<std::string>(item, TASK_REQUEST_ID_KEY);
    // ids.push_back(req_id);
    auto iter_req = req_status_.find(req_id);
    auto iter = item->find(TASK_MSG_KEY);
    if (iter != item->end()) {
      BatchInfo pro;
      pro.req_id = req_id;
      // prefill / finish
      auto re = dict_get<std::shared_ptr<TypedDict>>(item, TASK_MSG_KEY);
      parser_message(re, pro);

      // HAMI_ASSERT(pro.req_tokens + pro.max_tokens <= max_);
      HAMI_FATAL_ASSERT(
          pro.new_tokens == 0 &&
              pro.req_tokens + pro.max_tokens <=
                  page_size_ * page_table_->max_num_page(),
          "error input. page_table_->max_num_page()=" +
              std::to_string(page_table_->max_num_page()) + "\n");

      if (pro.finish) {
        // std::lock_guard<std::mutex> lock(req_status_mutex_);
        HAMI_FATAL_ASSERT(iter_req != req_status_.end(), "id=" + req_id);
        {
          iter_req->second.finish = true;
          iter_req->second.running = false;
          iter_req->second.data = item;
        }
        SPDLOG_INFO(
            "finishing `{}`, req_toks={}, new_toks={}",
            pro.req_id,
            iter_req->second.req_tokens,
            iter_req->second.new_tokens);
        continue;
      }

      HAMI_FATAL_ASSERT(iter_req == req_status_.find(pro.req_id));

      // SPDLOG_INFO("prefill: {} req_tokens={}", pro.req_id, pro.req_tokens);
      // page_table_->alloc(pro.req_id, pro.req_tokens);
      pro.data = item;
      // pro.event = dict_get<std::shared_ptr<Event>>(item, TASK_EVENT_KEY);
      static const auto start_time = helper::now();
      pro.time = helper::time_passed(start_time);
      req_status_.emplace(pro.req_id, std::move(pro));
      item->erase(iter);

    } else {
      // decode
      BatchInfo& pro = (iter_req->second);
      pro.new_tokens += 1;
      pro.running = false;
      // SPDLOG_INFO(
      //     "decoding: id = {}, req_toks={}, new_tokens = {}",
      //     pro.req_id,
      //     pro.req_tokens,
      //     pro.new_tokens);

      pro.data = item;
    }
  }

  for (auto iter = req_status_.begin(); iter != req_status_.end();) {
    if (iter->second.finish && !iter->second.running) {
      // SPDLOG_INFO("Contiguous Batching stoped: {}", iter->first);
      notify_event({iter->second.data});
      // iter->second.event->notify_all();
      SPDLOG_INFO("Contiguous Batching stoped(notify_event): {}", iter->first);
      HAMI_FATAL_ASSERT(page_table_->free(iter->first));
      iter = req_status_.erase(iter);
    } else {
      ++iter;
    }
  }

  // wait for all ready
  if (!std::all_of(
          req_status_.begin(), req_status_.end(), [](const auto& pair) {
            return !pair.second.running;
          })) {
    std::vector<std::string> not_ready_ids;
    for (const auto& pair : req_status_) {
      if (pair.second.running) {
        not_ready_ids.push_back(pair.first);
      }
    }
    SPDLOG_DEBUG(
        "contiguous batching: not all ready: " + str::join(not_ready_ids));
    return;
  }
  if (req_status_.empty())
    return;

  auto [activad_ids, finish_ids] = get_activated_ids();
  if (activad_ids.empty()) {
    SPDLOG_WARN(
        "returned. wired. empty ids. No memory or id?  num_ids= {}, available_pages = {} available_ids = {}",

        req_status_.size(),
        page_table_->available_pages(),
        page_table_->available_ids());
    return;
  }
  for (const auto& id : activad_ids) {
    // todo
    HAMI_FATAL_ASSERT(
        page_table_->alloc_or_reset(
            id, req_status_.at(id).req_tokens + req_status_.at(id).new_tokens),
        id + ": " +
            std::to_string(
                req_status_.at(id).req_tokens + req_status_.at(id).new_tokens) +
            " available pages=" +
            std::to_string(page_table_->available_pages()));
  }

  page_table_->activate(activad_ids);
  SPDLOG_INFO(
      "unactivated count={}; available pages={};activated({})=[{}]",
      req_status_.size() - activad_ids.size(),
      page_table_->available_pages(),
      activad_ids.size(),
      str::join(activad_ids));

  std::vector<dict> new_ios;
  for (const auto& id : activad_ids) {
    new_ios.emplace_back(req_status_.at(id).data);
    req_status_.at(id).running = true;
    auto iter = finish_ids.find(id);
    if (iter != finish_ids.end()) {
      new_ios.back()->insert({"finish_reason", iter->second});
      SPDLOG_INFO(
          "Contiguous Batching finish: {} finish_reason={}", id, iter->second);
      req_status_.at(id).finish = true;
    }
  }

  impl_forward_handle_except(new_ios, activad_ids);
  // SPDLOG_DEBUG(" {} finished one step.", str::join(activad_ids));
}

void ContiguousBatching::impl_forward_handle_except(
    const std::vector<dict>& ios,
    const std::vector<id_type>& ids) {
  std::vector<std::shared_ptr<Event>> events;
  for (const auto& item : ios) {
    auto iter = item->find(TASK_EVENT_KEY);
    events.push_back(any_cast<std::shared_ptr<Event>>(iter->second));
    item->erase(iter);
    item->erase(TASK_RESULT_KEY);
  }
  try {
    dependency_->forward(ios);
  } catch (...) {
    for (std::size_t i = 0; i < ios.size(); ++i) {
      (*ios[i])[TASK_EVENT_KEY] = events[i];
      ios[i]->erase(TASK_RESULT_KEY);
    }
    for (const auto& ev : events) {
      ev->set_exception_and_notify_all(std::current_exception());
    }
    for (const auto& id : ids) {
      // req_status_.at(id).running = false;
      req_status_.erase(id);
      page_table_->free(id);
    }
    page_table_->deactivate();
    try {
      std::rethrow_exception(std::current_exception());
    } catch (std::exception& e) {
      SPDLOG_ERROR("batching error: {}", e.what());
    }
    return;
  }
  for (std::size_t i = 0; i < ios.size(); ++i) {
    (*ios[i])[TASK_EVENT_KEY] = events[i];
  }
  for (const auto& ev : events) {
    ev->notify_all();
  }
}

void ContiguousBatching::parser_message(
    const std::shared_ptr<TypedDict>& msg,
    BatchInfo& pro) {
  pro.finish = try_get<std::string>(*msg, "action") == "finish";
  if (pro.finish)
    return;
  pro.req_tokens = get<int>(*msg, "req_tokens");
  pro.context_length = get<int>(*msg, "context_length");
  try_update<int>(*msg, "max_tokens", pro.max_tokens);

  // if (pro.context_length == 0)
  //   pro.context_length = std::numeric_limits<int>::max();
  // if (pro.max_tokens == 0)
  //   pro.max_tokens = std::numeric_limits<int>::max();
  HAMI_FATAL_ASSERT(pro.context_length > 0 || pro.max_tokens > 0);

  SPDLOG_INFO(
      "\n"
      "+---------------------------- Contiguous Batching ----------------------------+\n"
      "| Request ID:      {:45} |\n"
      "| Req Tokens:      {:45} |\n"
      "| Context Length:      {:45} |\n"
      "| Max (New) Tokens:  {:45} |\n"
      "+------------------------------------------------------------------------------+",
      pro.req_id,
      pro.req_tokens,
      pro.context_length,
      pro.max_tokens);
}

HAMI_REGISTER_BACKEND(ContiguousBatching);

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
  // auto inputs = inputs_data;
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
