#include <numeric>
#include <shared_mutex>
#include <unordered_map>
// #include <shared_lock>

#include "omniback/builtin/page_table.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/timer.hpp"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/extra/stl.h>

namespace omniback {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // using omniback::ffi::Any;
  refl::ObjectDef<PageTable::PageInfo>()
      .def(refl::init<>())
      // .def_readwrite("kv_page_indices",
      // &PageTable::PageInfo::kv_page_indices)
      .def_rw("kv_page_indices", &PageTable::PageInfo::kv_page_indices)
      .def_rw("kv_last_page_len", &PageTable::PageInfo::kv_last_page_len);

  refl::ObjectDef<PageTable>()
      // Constructors
      .def(refl::init<>())
      // .def(refl::init<size_t, size_t, size_t>()) // max_num_req, max_num_page,
                                                 // page_size
      .def(
          "init",
          [](PageTable* self,
             size_t max_num_req,
             size_t max_num_page,
             size_t page_size) -> PageTable* {
            self->init(max_num_req, max_num_page, page_size);
            return self; // 返回 self 实现链式调用
          })
      .def(
          "alloc",
          &PageTable::alloc) // id, num_tok
      .def("reset", &PageTable::reset)
      .def("extend", &PageTable::extend)
      .def("free", &PageTable::free)
      .def(
          "page_table",
          [](PageTable* self, const std::vector<id_type>& id) {
            std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> re;
            {
              return re = self->page_table(id);
            }
          })
      .def("add_more_page", &PageTable::add_more_page)
      .def("available_pages", &PageTable::available_pages)
      .def("get_num_tok", &PageTable::get_num_tok)
      .def(
          "get_prefill_size",
          [](PageTable* self, const std::vector<id_type>& ids) {
            return (self->get_prefill_size(ids));
          })
      .def(
          "get_current_size",
          [](PageTable* self, const std::vector<id_type>& ids) {
            return (self->get_current_size(ids));
          })
      .def("available_ids", &PageTable::available_ids)
      // .def("pop_activated", &PageTable::pop_activated)
      .def(
          "pop_activated",
          [](PageTable* self) {
            auto re= self->pop_activated();
            return std::make_tuple(re.first, re.second);
          })
      .def("page_info", [](const PageTable* self, const id_type& id) {
        return &(self->page_info(id));
      });
  ;
}

bool PageTable::reset(const omniback::id_type& name, size_t num_tok) {
  // std::lock_guard<std::mutex> lock(page_infos_lock_);
  // auto iter = page_infos_.find(name);
  // OMNI_ASSERT(iter != page_infos_.end());
  // auto& info = iter->second;
  const auto total = get_num_tok(name);
  if (total >= num_tok) {
    SPDLOG_WARN("extend: total >= num_tok");
    return true;
  }
  // SPDLOG_INFO(
  //     "PageTable::reset(tokens): id={},now={},required={}",
  //     name,
  //     total,
  //     num_tok);

  std::lock_guard<std::mutex> lock(page_infos_lock_);
  auto& info = *page_infos_.at(name);

  if (page_size_ - info.kv_last_page_len >= num_tok - total) {
    info.kv_last_page_len += num_tok - total;
    return true;
  }
  //   PageInfo new_info;
  auto need_new_tok = num_tok - (page_size_ - info.kv_last_page_len + total);
  auto kv_page_indices =
      slots_.alloc((need_new_tok + page_size_ - 1) / page_size_);
  if (kv_page_indices.empty()) {
    SPDLOG_WARN(
        "slots_.alloc failed. num_tok={}, need_new_tok={}, total={}",
        num_tok,
        need_new_tok,
        total);
    return false;
  }
  info.kv_page_indices.insert(
      info.kv_page_indices.end(),
      kv_page_indices.begin(),
      kv_page_indices.end());
  info.kv_last_page_len = (need_new_tok % page_size_) == 0
      ? need_new_tok
      : (need_new_tok % page_size_);
  return true;
}

float PageTable::get_time() {
  static const auto start_time = helper::now();
  return helper::time_passed(start_time);
}

bool PageTable::extend(const omniback::id_type& name) {
  // std::lock_guard<std::mutex> lock(page_infos_lock_);
  // auto iter = page_infos_.find(name);
  // OMNI_ASSERT(iter != page_infos_.end());
  // auto& info = iter->second;
  // const auto total = get_num_tok(name);

  std::lock_guard<std::mutex> lock(page_infos_lock_);
  auto& info = *page_infos_.at(name);

  if (page_size_ - info.kv_last_page_len >= 1) {
    info.kv_last_page_len += 1;
    return true;
  }
  //   PageInfo new_info;
  auto kv_page_indices = slots_.alloc(1);
  if (kv_page_indices.empty()) {
    return false;
  }
  info.kv_page_indices.insert(
      info.kv_page_indices.end(),
      kv_page_indices.begin(),
      kv_page_indices.end());
  info.kv_last_page_len = 1;
  return true;
}

std::pair<std::vector<id_type>, std::vector<int>> PageTable::pop_activated() {
  std::pair<std::vector<id_type>, std::vector<int>> re;

  std::lock_guard<std::mutex> lock(page_infos_lock_);
  if (ids_.empty())
    return re;
  // re.first = ids_.front();
  std::swap(ids_.front(), re.first);
  ids_.pop();
  for (const id_type& id : re.first) {
    auto iter = page_infos_.find(id);
    OMNI_FATAL_ASSERT(
        iter != page_infos_.end(),
        id + " not found. Size = " + std::to_string(page_infos_.size()) +
            " name = " + page_infos_.begin()->first);
    const auto& item = *(iter->second);
    OMNI_FATAL_ASSERT(!item.kv_page_indices.empty());

    re.second.push_back(
        item.kv_last_page_len + page_size_ * (item.kv_page_indices.size() - 1));
  }

  return re;
}

std::vector<int> PageTable::get_current_size(const std::vector<id_type>& ids) {
  std::vector<int> re;
  std::lock_guard<std::mutex> lock(page_infos_lock_);

  for (const id_type& id : ids) {
    auto iter = page_infos_.find(id);
    OMNI_FATAL_ASSERT(
        iter != page_infos_.end(),
        id + " not found. Size = " + std::to_string(page_infos_.size()) +
            " name = " + page_infos_.begin()->first);
    const auto& item = *(iter->second);

    int total =
        item.kv_last_page_len + page_size_ * (item.kv_page_indices.size() - 1);
    if (total == item.init_size)
      re.push_back(total);
    else
      re.push_back(total - item.init_size);
  }

  return re;
}

std::vector<int> PageTable::get_prefill_size(const std::vector<id_type>& ids) {
  std::vector<int> re;
  std::lock_guard<std::mutex> lock(page_infos_lock_);

  for (const id_type& id : ids) {
    auto iter = page_infos_.find(id);
    OMNI_FATAL_ASSERT(
        iter != page_infos_.end(),
        id + " not found. Size = " + std::to_string(page_infos_.size()) +
            " name = " + page_infos_.begin()->first);
    const auto& item = *(iter->second);

    re.push_back(item.init_size);
  }

  return re;
}

std::pair<std::vector<id_type>, std::vector<int>> PageTable::get_activated() {
  std::pair<std::vector<id_type>, std::vector<int>> re;

  std::lock_guard<std::mutex> lock(page_infos_lock_);
  if (ids_.empty())
    return re;
  re.first = ids_.front();
  // std::swap(ids_.front(), re.first);
  // ids_.pop();
  for (const id_type& id : re.first) {
    auto iter = page_infos_.find(id);
    OMNI_FATAL_ASSERT(
        iter != page_infos_.end(),
        id + " not found. Size = " + std::to_string(page_infos_.size()) +
            " name = " + page_infos_.begin()->first);
    const auto& item = *(iter->second);
    OMNI_FATAL_ASSERT(!item.kv_page_indices.empty());

    re.second.push_back(
        item.kv_last_page_len + page_size_ * (item.kv_page_indices.size() - 1));
  }

  return re;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> PageTable::
    page_table(const std::vector<id_type>& id) {
  // std::vector<int> kv_page_indices;
  // std::vector<int> kv_page_indptr;
  // std::vector<int> kv_last_page_len;

  // OMNI_ASSERT(id.size() == seq_lens.size(0) && seq_lens.is_cpu());
  size_t total{0};

  std::lock_guard<std::mutex> lock(page_infos_lock_);
  for (size_t i = 0; i < id.size(); ++i) {
    total += page_infos_.at(id[i])->kv_page_indices.size();
  }

  std::vector<int> kv_page_indices;
  kv_page_indices.reserve(total);

  std::vector<int> kv_page_indptr(1 + id.size(), 0);
  std::vector<int> kv_last_page_len(id.size());
  for (size_t i = 0; i < id.size(); ++i) {
    const auto& infor = *(page_infos_.at(id[i]));
    kv_page_indices.insert(
        kv_page_indices.end(),
        infor.kv_page_indices.begin(),
        infor.kv_page_indices.end());
    kv_page_indptr[i + 1] = kv_page_indptr[i] + infor.kv_page_indices.size();
    kv_last_page_len[i] = infor.kv_last_page_len;
  }

  return std::make_tuple(kv_page_indices, kv_page_indptr, kv_last_page_len);
}

void PageTable::activate(std::vector<id_type> ids) {
  std::lock_guard<std::mutex> lock(page_infos_lock_);
  for (const auto& item : ids) {
    OMNI_ASSERT(page_infos_.find(item) != page_infos_.end(), "id=" + item);
  }
  ids_.push(std::move(ids));
}

void PageTable::deactivate() {
  std::lock_guard<std::mutex> lock(page_infos_lock_);
  ids_ = {};
  // decltype(ids_) empty;
  // std::swap(ids_, empty);
}

// bool PageTable::alloc_pages(const omniback::id_type& name, size_t num_page) {
//   std::unique_lock<std::mutex> lock(page_infos_lock_);

//   auto& info = page_infos_.at(name);

//   if (page_size_ - info.kv_last_page_len >= num_tok - total) {
//     info.kv_last_page_len += num_tok - total;
//     return true;
//   }
//   //   PageInfo new_info;
//   auto need_new_tok = num_tok - (page_size_ - info.kv_last_page_len + total);
//   auto kv_page_indices =
//       page_table_.alloc((need_new_tok + page_size_ - 1) / page_size_);
//   if (kv_page_indices.empty()) {
//     return false;
//   }
//   info.kv_page_indices.insert(
//       info.kv_page_indices.end(),
//       kv_page_indices.begin(),
//       kv_page_indices.end());
//   info.kv_last_page_len = (need_new_tok % page_size_) == 0
//       ? need_new_tok
//       : (need_new_tok % page_size_);
//   return true;
// }

bool PageTable::alloc_or_reset(const omniback::id_type& name, size_t num_tok) {
  bool find_id = false;
  {
    std::unique_lock<std::mutex> lock(page_infos_lock_);
    find_id = page_infos_.find(name) != page_infos_.end();
  }
  // SPDLOG_INFO(
  //     "PageTable::alloc_or_reset(tokens): id={},find_id={},num_tok={}",
  //     name,
  //     find_id,
  //     num_tok);
  // todo: lock
  if (find_id)
    return reset(name, num_tok);
  else
    return alloc(name, num_tok);
}

bool PageTable::alloc(const omniback::id_type& name, size_t num_tok) {
  std::unique_lock<std::mutex> lock(page_infos_lock_);
  if (page_infos_.size() >= max_num_req_) {
    SPDLOG_WARN(
        "page_infos_.size(){} >= max_num_req_ {}",
        page_infos_.size(),
        max_num_req_);
    return false;
  }

  OMNI_ASSERT(page_infos_.find(name) == page_infos_.end());

  auto [iter, inserted] = page_infos_.emplace(name, PageTable::empty_info());
  if (!inserted)
    return false;
  auto& info = *(iter->second);

  info.kv_page_indices = slots_.alloc((num_tok + page_size_ - 1) / page_size_);
  info.kv_last_page_len =
      (num_tok % page_size_) == 0 ? page_size_ : (num_tok % page_size_);
  info.init_size = num_tok;
  // SPDLOG_INFO("Alloc {}, {}", name, num_tok);
  info.time = get_time();
  if (info.kv_page_indices.empty()) {
    page_infos_.erase(name);
    SPDLOG_WARN("slots_.alloc failed. num_tok={}", num_tok);
    return false;
  }

  return true;
}

PageTable& default_page_table(const std::string& tag) {
  static std::shared_mutex map_mutex;
  static std::unordered_map<std::string, tvm::ffi::ObjectPtr<PageTable>>
      page_table_map;

  {
    std::shared_lock lock(map_mutex);
    auto it = page_table_map.find(tag);
    if (it != page_table_map.end()) {
      return *(it->second.get());
    }
  }

  std::unique_lock lock(map_mutex);
  auto& ptr = page_table_map[tag];
  if (!ptr) {
    ptr = tvm::ffi::make_object<PageTable>();
  }
  return *(ptr.get());
}
PageTable* py_default_page_table(const std::string& tag) {
  return &(default_page_table(tag));
}
  
  
TVM_FFI_DLL_EXPORT_TYPED_FUNC(default_page_table, py_default_page_table);

} // namespace omniback
