#include "omniback/schedule/continuous_batching.hpp"

#include <charconv>

#include "omniback/builtin/aspect.hpp"
#include "omniback/builtin/proxy.hpp"
#include "omniback/core/event.hpp"
#include "omniback/core/helper.hpp"
#include "omniback/core/parser.hpp"
#include "omniback/core/queue.hpp"
#include "omniback/core/reflect.h"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/string.hpp"
#include "omniback/helper/timer.hpp"

namespace omniback {
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

void PlainContinuousBatching::impl_init(
    const std::unordered_map<string, string>& params,
    const dict& options) {
  auto [args, kwargs] =
      parser_v2::get_args_kwargs(this, "PlainContinuousBatching", params);
  std::string target = str::get<std::string>(kwargs, "target");
  // max_ = str::get<int>(kwargs, "max");
  // auto no_page_table = str::get(kwargs, "no_page_table");
  // no_page_table_ = init_backend(no_page_table, params, options);
  SPDLOG_INFO("plain contiguous batching, target = {}", target);

  dependency_ = OMNI_INSTANCE_GET(Backend, target);
  OMNI_ASSERT(dependency_, target + " not found (PlainContinuousBatching).");
  if (args.size() >= 1) {
    src_queue_ = &(default_queue(args[0]));
    SPDLOG_INFO("PlainContinuousBatching USE QUEUE: {}", args[0]);
  } else {
    SPDLOG_INFO("USE QUEUE: PlainContinuousBatching::Queue");
    src_queue_ = &(default_queue("PlainContinuousBatching::Queue"));
  }

  thread_ = std::thread(&PlainContinuousBatching::task_loop, this);
} // 异常 错误 共识

void PlainContinuousBatching::task_loop() {
  // while (bInited_.load()) {
  //   // only one thread runing this loop
  //   dict tmp_data = nullptr;
  //   if (!src_queue_->wait_pop(tmp_data, 100)) {
  //     continue;
  //   }
  //   std::string id = dict_get<std::string>(tmp_data, TASK_DATA_KEY);
  //   auto time_now =
  //       std::chrono::duration<float>(
  //           std::chrono::system_clock::now().time_since_epoch())
  //           .count();
  //   auto ev = dict_get<Event>(tmp_data, TASK_EVENT_KEY);
  //   tmp_data->erase(TASK_EVENT_KEY);
  //   receiving_data_[id] = TaskInfo({id, tmp_data, time_now, ev, 0, 0});
  //   if (!all_received()){
  //     continue;
  //   }

  //   for (auto& rec : receiving_data_) {
  //     if (!cached_data_.contains(rec.first)){
  //       cached_data_[rec.first] = rec.second;
  //     }else{
  //       rec.second.loop_index = cached_data_[rec.first].loop_index + 1;
  //       rec.second.delay = cached_data_[rec.first].delay;
  //       std::swap(cached_data_[rec.first], rec.second);
  //     }
  //   }
  //   receiving_data_.clear();
  //   std::vector<dict> datas ;
  //   for (const auto& item : cached_data_){
  //     auto data = make_dict();
  //     (*data)["data"] = item.first;
  //     (*data)["time"] = item.second.time;
  //     datas.push_back(data);
  //   }
  //   dependency_->forward(datas);
  // }
}

void PlainContinuousBatching::impl_forward(const std::vector<dict>& io) {}

void ContinuousBatching::impl_init(
    const std::unordered_map<string, string>& params,
    const dict& options) {
  auto [args, kwargs] =
      parser_v2::get_args_kwargs(this, "ContinuousBatching", params);
  std::string target = str::get<std::string>(kwargs, "target");
  max_ = str::get<int>(kwargs, "max");
  // auto no_page_table = str::get(kwargs, "no_page_table");
  // no_page_table_ = init_backend(no_page_table, params, options);
  SPDLOG_INFO("contiguous batching, target = {}, max = {}", target, max_);

  dependency_ = OMNI_INSTANCE_GET(Backend, target);
  OMNI_ASSERT(dependency_, target + " not found (ContinuousBatching).");

  page_table_ = &default_page_table();
  page_size_ = page_table_->page_size();
  OMNI_ASSERT(page_size_ > 0);
}
std::pair<std::vector<id_type>, std::unordered_map<id_type, std::string>>
ContinuousBatching::get_activated_ids() {
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
      OMNI_FATAL_ASSERT(max_token > 0 && max_token < INT_MAX);
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
  // limited by `max`(max batch size of the network) for prefill (prefill
  // first for batch size limitation)
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
    OMNI_ASSERT(prefill_ids.empty());
    {
      // parse will_finish_ids
      stable_sort_by_time(will_finish_ids);
      for (auto iter = will_finish_ids.begin();
           iter != will_finish_ids.end();) {
        if (new_page_needed[iter->first] == 0) {
          iter++;
          continue;
        }
        OMNI_FATAL_ASSERT(
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
        OMNI_FATAL_ASSERT(new_page_needed[*iter] == 1);
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

  // Controls Droping mechanism for unfinished inference tasks
  // (wont_finish_ids) when insufficient memory pages exist for future
  // processing.

  /* Drop Strategy Options:
   * -------------------------
   * [Trigger Condition] When should we drop (offload to CPU or trigger
   * re-computation)?
   *
   * Option A: drop one when available_pages == 0 for the next round
   *
   * Option B: drop one when dropped decode ids(because of no pages) > 0.4
   * of total decodes
   */
  if (will_finish_ids.empty() && prefill_ids.empty()) {
    OMNI_FATAL_ASSERT(
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

void ContinuousBatching::impl_forward(const std::vector<dict>& io) {
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

      // OMNI_ASSERT(pro.req_tokens + pro.max_tokens <= max_);
      OMNI_FATAL_ASSERT(
          pro.new_tokens == 0 &&
              pro.req_tokens + pro.max_tokens <=
                  page_size_ * page_table_->max_num_page(),
          "error input. page_table_->max_num_page()=" +
              std::to_string(page_table_->max_num_page()) + "\n");

      if (pro.finish) {
        // std::lock_guard<std::mutex> lock(req_status_mutex_);
        OMNI_FATAL_ASSERT(iter_req != req_status_.end(), "id=" + req_id);
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

      OMNI_FATAL_ASSERT(iter_req == req_status_.find(pro.req_id));

      // SPDLOG_INFO("prefill: {} req_tokens={}", pro.req_id,
      // pro.req_tokens); page_table_->alloc(pro.req_id, pro.req_tokens);
      pro.data = item;
      // pro.event = dict_get<Event>(item, TASK_EVENT_KEY);
      if (pro.time <= 0)
        pro.time = helper::timestamp();
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
      notify_event({iter->second.data});
      // iter->second.event->notify_all();
      SPDLOG_INFO("Continuous Batching stoped(notify_event): {}", iter->first);
      OMNI_FATAL_ASSERT(page_table_->free(iter->first));
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
    OMNI_FATAL_ASSERT(
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
          "Continuous Batching finish: {} finish_reason={}", id, iter->second);
      req_status_.at(id).finish = true;
    }
  }

  impl_forward_handle_except(new_ios, activad_ids);
  // SPDLOG_DEBUG(" {} finished one step.", str::join(activad_ids));
}

void ContinuousBatching::impl_forward_handle_except(
    const std::vector<dict>& ios,
    const std::vector<id_type>& ids) {
  // We remove the event here to enforce synchronous semantics.
  // Alternatively, asynchronous semantics can be maintained by setting the
  // callback: ev->set_exception_callback()
  std::vector<Event> events;
  for (const auto& item : ios) {
    auto iter = item->find(TASK_EVENT_KEY);
    events.push_back(any_cast<Event>(iter->second));
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

void ContinuousBatching::parser_message(
    const std::shared_ptr<TypedDict>& msg,
    BatchInfo& pro) {
  pro.finish = try_get<std::string>(*msg, "action") == "finish";
  if (pro.finish)
    return;
  pro.req_tokens = get<int>(*msg, "req_tokens");
  pro.context_length = get<int>(*msg, "context_length");
  try_update<int>(*msg, "max_tokens", pro.max_tokens);
  try_update<double>(*msg, "timestamp", pro.time);

  // if (pro.context_length == 0)
  //   pro.context_length = std::numeric_limits<int>::max();
  // if (pro.max_tokens == 0)
  //   pro.max_tokens = std::numeric_limits<int>::max();
  OMNI_FATAL_ASSERT(pro.context_length > 0 || pro.max_tokens > 0);

  SPDLOG_INFO(
      "\n"
      "+---------------------------- Continuous Batching ----------------------------+\n"
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

OMNI_REGISTER_BACKEND(ContinuousBatching);

} // namespace omniback
