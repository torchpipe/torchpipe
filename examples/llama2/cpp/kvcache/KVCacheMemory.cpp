#include <thread>
#include "KVCacheMemory.hpp"
// #include "torch/torch.h"
#include "Memory.hpp"
#include "ipipe_common.hpp"
#include "base_logging.hpp"

namespace {}  // namespace
namespace kvcache {

void CachedMemory::offload2cpu(size_t seq_len_with_output) {
  IPIPE_ASSERT(cpu_ptr_ == nullptr);

  size_t pitch = config_.elemsize * config_.max_seq_len * config_.hidden_size;
  cpu_ptr_ = alloc_pinned(seq_len_with_output * config_.elemsize * config_.hidden_size *
                          config_.layer_num);
  size_t w = seq_len_with_output * config_.elemsize * config_.hidden_size;

  pyh_pool_->offload_memcpy2d(cpu_ptr_, ptrs_[0], w, config_.layer_num, pitch);

  repair_by_unmap(0);
  // DRV_CALL(cuMemcpyDtoHAsync(cpu_ptrs_[0], reinterpret_cast<CUdeviceptr>(ptrs_[0]),
  //                            config_.granularitySize * config_.layer_num, 0));
};

void CachedMemory::onload2gpu(size_t seq_len_with_output, size_t target_blk_len) {
  repair_by_map(target_blk_len);
  IPIPE_ASSERT(cpu_ptr_ != nullptr);

  size_t pitch = config_.elemsize * config_.max_seq_len * config_.hidden_size;

  size_t w = seq_len_with_output * config_.elemsize * config_.hidden_size;

  pyh_pool_->onload_memcpy2d(ptrs_[0], cpu_ptr_, w, config_.layer_num, pitch);

  free_pinned(cpu_ptr_);
  cpu_ptr_ = nullptr;

  // DRV_CALL(cuMemcpyDtoHAsync(cpu_ptrs_[0], reinterpret_cast<CUdeviceptr>(ptrs_[0]),
  //                            config_.granularitySize * config_.layer_num, 0));
};

void KVCacheMemory::init(const KVCacheConfig& config) {
  config_ = config;
  pyh_pool_ = std::make_unique<PyhBlkPool>(config.device_id, config_.granularitySize);

  extend(config.max_concurrent_requests * 2);  // for reserved

  seq_per_block_ = config.granularitySize / (config.hidden_size * config.elemsize);

  thread_ = std::thread(&KVCacheMemory::task_loop, this);
}

int KVCacheMemory::get_reserved_blocks() {
  int reserve = 0;
  for (const auto& mem_state : memory_state_) {
    const auto owned_block_len =
        cached_memories_[mem_state.second.memory_index * 2]->get_block_len();
    if (owned_block_len > mem_state.second.target_block_len) {
      reserve += owned_block_len - mem_state.second.target_block_len;
    }
  }
  return reserve * 2 * config_.layer_num;
}
// void KVCacheMemory::alloc_reqid(const KVCacheAllocParams& request_id) {
//   // Implementation of the alloc_reqid function
// }
void KVCacheMemory::step(const CachedMemoryTasks& tasks) {
  if (!tasks.onload2gpu.empty()) {
    auto func = [this, onload2gpu = tasks.onload2gpu]() {
      for (const auto& req : onload2gpu) {
        cached_memories_[memory_state_[req].memory_index * 2]->onload2gpu(
            memory_state_[req].seq_len_with_output, memory_state_[req].target_block_len);
        cached_memories_[1 + memory_state_[req].memory_index * 2]->onload2gpu(
            memory_state_[req].seq_len_with_output, memory_state_[req].target_block_len);
        memory_state_[req].offloaded = false;
      }
    };
    task_queue_.Push(func);
  }
  if (!tasks.offload2cpu.empty()) {
    auto func = [this, offload2cpu = tasks.offload2cpu]() {
      for (const auto& req : offload2cpu) {
        memory_state_[req].offloaded = true;
        cached_memories_[memory_state_[req].memory_index * 2]->offload2cpu(
            memory_state_[req].seq_len_with_output);
        cached_memories_[1 + memory_state_[req].memory_index * 2]->offload2cpu(
            memory_state_[req].seq_len_with_output);
      }
    };
    task_queue_.Push(func);
  }

  if (tasks.alloc_blks > 0) {
    auto func = [this, &tasks]() { pyh_pool_->alloc(tasks.alloc_blks); };
    // pyh_pool_->alloc(tasks.alloc_blks);
    task_queue_.Push(func);
  }
  if (tasks.free_reserved > 0) {
    auto func = [this, &tasks]() { free_reserved(tasks.free_reserved); };
    task_queue_.Push(func);
  }

  return;
}

void CachedMemory::release() {
  IPIPE_ASSERT(repair_by_unmap(0));
  // for (auto ptr : ptrs_) {
  //   virtual_free(ptr, context_max_size_ * config_.layer_num);
  // }
  virtual_free(ptrs_[0], context_max_size_ * config_.layer_num);
  if (cpu_ptr_) {
    free_pinned(cpu_ptr_);
  }
  cpu_ptr_ = nullptr;
  ptrs_.clear();
}

bool CachedMemory::repair_by_unmap(size_t needed_blk) {
  if (phy_blocks_[0].size() < needed_blk) {
    return false;
  }
  SPDLOG_INFO("repair_by_unmap: before: {}, after: {}", phy_blocks_[0].size(), needed_blk);
  for (size_t i = 0; i < config_.layer_num; i++) {
    for (size_t j = phy_blocks_[i].size() - 1; j >= needed_blk; j--) {
      phy_blocks_[i][j]->unmap(ptrs_[i] + (j)*config_.granularitySize);
      pyh_pool_->free(phy_blocks_[i][j]);
    }
    phy_blocks_[i].resize(needed_blk);
  }
  return true;
}

bool CachedMemory::repair_by_map(size_t needed_blk) {
  size_t blk = phy_blocks_[0].size();
  if ((needed_blk < phy_blocks_[0].size()) ||
      (needed_blk - phy_blocks_[0].size()) * config_.layer_num > pyh_pool_->size()) {
    return false;
  }
  for (size_t i = 0; i < config_.layer_num; i++) {
    while (phy_blocks_[i].size() < needed_blk) {
      auto blk = pyh_pool_->get_free_blk();
      IPIPE_ASSERT(blk);
      phy_blocks_[i].push_back(blk);
      phy_blocks_[i].back()->map(ptrs_[i] + (phy_blocks_[i].size() - 1) * config_.granularitySize);
    }
  }
  SPDLOG_INFO("repair_by_map: before: {}, after: {}", blk, phy_blocks_[0].size());
  return true;
}

// void KVCacheMemory::alloc(const std::unordered_set<std::string>& prefill_alloc_reqs,
//                           const std::unordered_set<std::string>& decode_alloc_reqs);

size_t KVCacheMemory::get_best_match(int blk_num) {
  int min_abs = INT32_MAX;
  size_t re{0};
  for (const size_t index : reserved_memory_index_) {
    int cur = std::abs(int(cached_memories_[2 * index]->get_block_len()) - blk_num);
    if (cur < min_abs) {
      min_abs = cur;
      re = index;
    }
  }
  return re;
}

void KVCacheMemory::alloc(std::unordered_set<std::string> prefill_alloc_reqs,
                          const std::map<std::string, KVCacheState>& prefill_status,
                          std::unordered_set<std::string> decode_alloc_reqs,
                          const std::map<std::string, KVCacheState>& decode_status) {
  for (const auto& prefill_alloc_req : prefill_alloc_reqs) {
    memory_state_.emplace(prefill_alloc_req, KVCacheMemoryState());
    auto& mem_state = memory_state_[prefill_alloc_req];

    if (!reserved_memory_index_.empty()) {
      mem_state.seq_len_with_output = prefill_status.at(prefill_alloc_req).seq_len_with_output;
      mem_state.target_block_len = get_target_block_len(mem_state.seq_len_with_output);
      size_t index = get_best_match(mem_state.target_block_len);
      // reserved_memory_index_.erase(index);
      mem_state.memory_index = index;

      // owned_block_len = cached_memories_[2 * index]->get_block_len();
      reserved_memory_index_.erase(index);
      // mem_state.target_seq_len = prefill_status[prefill_alloc_req].seq_len_with_output;
      // cached_memories_[2 * index].set_target_seq_len(
      //     prefill_status[prefill_alloc_req].seq_len_with_output);
      // cached_memories_[1 + 2 * index].set_target_seq_len(
      //     prefill_status[prefill_alloc_req].seq_len_with_output);
      // need_repair_memory_index_.insert(index);
    } else {
      if (free_memory_index_.empty()) {
        extend(config_.max_concurrent_requests / 4 + 1);
      }

      auto iter = free_memory_index_.begin();
      // need_repair_memory_index_.insert(*iter);
      // cached_memories_[2 * (*iter)].set_target_seq_len(
      //     prefill_status[prefill_alloc_req].seq_len_with_output);
      // cached_memories_[1 + 2 * (*iter)].set_target_seq_len(
      //     prefill_status[prefill_alloc_req].seq_len_with_output);
      // set_target_seq_len(index, prefill_status[prefill_alloc_req].seq_len_with_output);
      // mem_state.memory_index = *iter;
      // mem_state.target_seq_len = prefill_status[prefill_alloc_req].seq_len_with_output;

      mem_state.memory_index = *iter;
      mem_state.seq_len_with_output = prefill_status.at(prefill_alloc_req).seq_len_with_output;

      mem_state.target_block_len = get_target_block_len(mem_state.seq_len_with_output);
      // mem_state.owned_block_len = cached_memories_[2 * (*iter)]->get_block_len();
      free_memory_index_.erase(iter);
    }
    SPDLOG_INFO("prefill_alloc_req: {}, target={}, memory_index={}", prefill_alloc_req,
                mem_state.target_block_len, mem_state.memory_index);
  }

  for (const auto& decode_alloc_req : decode_alloc_reqs) {
    // std::lock_guard<std::mutex> lock(memory_state_mutex_);

    auto& mem_state = memory_state_[decode_alloc_req];
    IPIPE_ASSERT(!mem_state.offloaded);
    mem_state.target_block_len =
        get_target_block_len(decode_status.at(decode_alloc_req).seq_len_with_output);

    SPDLOG_INFO("decode_alloc_req: {}, target={}, memory_index={}", decode_alloc_req,
                mem_state.target_block_len, mem_state.memory_index);
  }
  repair();
  // repair_by_new_map();
}

const MemOutput& KVCacheMemory::query_key(const std::string& request_id) {
  auto iter = memory_state_.find(request_id);
  IPIPE_ASSERT(iter != memory_state_.end());
  return cached_memories_[iter->second.memory_index * 2]->get_ptrs();
}
const MemOutput& KVCacheMemory::query_value(const std::string& request_id) {
  auto iter = memory_state_.find(request_id);
  IPIPE_ASSERT(iter != memory_state_.end());
  return cached_memories_[1 + iter->second.memory_index * 2]->get_ptrs();
}
int KVCacheMemory::free_reserved(size_t blk_size) {
  int need_new_blk_len = blk_size;  //, reserved_used_blk_len = 0, reserved_blk_len = 0;

  if (need_new_blk_len > 0) {  // part used
    for (const auto& iter : memory_state_) {
      const auto& mem_state = iter.second;
      const auto owned_block_len = cached_memories_[2 * mem_state.memory_index]->get_block_len();
      if (owned_block_len > mem_state.target_block_len && (mem_state.target_block_len != 0)) {
        cached_memories_[mem_state.memory_index * 2]->repair_by_unmap(mem_state.target_block_len);
        cached_memories_[1 + mem_state.memory_index * 2]->repair_by_unmap(
            mem_state.target_block_len);
        need_new_blk_len -= 2 * config_.layer_num * (owned_block_len - mem_state.target_block_len);
        if (need_new_blk_len <= 0) break;
      }
    }
  }
  if (need_new_blk_len > 0) {  // reserve and not used
    for (auto iter = memory_state_.begin(); iter != memory_state_.end();) {
      const auto& mem_state = iter->second;
      // if (mem_state.offloaded) continue;
      const auto owned_block_len = cached_memories_[2 * mem_state.memory_index]->get_block_len();

      if (owned_block_len > 0 && mem_state.target_block_len == 0) {
        IPIPE_ASSERT(reserved_memory_index_.count(mem_state.memory_index) != 0);
        cached_memories_[mem_state.memory_index * 2]->repair_by_unmap(0);
        cached_memories_[1 + mem_state.memory_index * 2]->repair_by_unmap(0);
        free_memory_index_.insert(mem_state.memory_index);
        reserved_memory_index_.erase(mem_state.memory_index);
        iter = memory_state_.erase(iter);
        need_new_blk_len -= 2 * config_.layer_num * owned_block_len;
        if (need_new_blk_len <= 0) break;
      } else {
        iter++;
      }
    }
  }
  return need_new_blk_len;
}

void KVCacheMemory::task_loop() {
  while (bInited_.load()) {
    std::function<void()> task;
    if (task_queue_.WaitForPop(task, 300)) {
      task();
    }
  }

  release();

  // try {
  //   release();
  // } catch (const std::exception& e) {
  //   SPDLOG_ERROR("KVCacheMemory release error. Memory leak may occur: {}", e.what());
  // }
}

void KVCacheMemory::cache(const CacheParams& ca) {
  // if (ca.event) {
  //   // IPIPE_ASSERT(cudaStreamWaitEvent(stream_, ca.event) == cudaSuccess);
  //   DRV_CALL(cuStreamWaitEvent(stream_, (CUevent)ca.event, 0));
  // }
  // prefill 需要搬运； decode不需要搬运！
}

void KVCacheMemory::repair() {
  SPDLOG_INFO("before repair: free={},reserved={}", get_free_blocks(), get_reserved_blocks());
  int need_new_blk_len = 0;  //, reserved_used_blk_len = 0, reserved_blk_len = 0;

  for (const auto& mem_state : memory_state_) {
    const auto owned_block_len =
        cached_memories_[mem_state.second.memory_index * 2]->get_block_len();
    if (owned_block_len < mem_state.second.target_block_len) {
      need_new_blk_len +=
          2 * config_.layer_num * (mem_state.second.target_block_len - owned_block_len);
    }
  }
  SPDLOG_INFO("repair:  need_new_blk_len={}", need_new_blk_len);
  if (need_new_blk_len > get_free_blocks()) {
    need_new_blk_len = free_reserved(need_new_blk_len - get_free_blocks());

    if (need_new_blk_len > 0) {
      SPDLOG_ERROR(
          "KVCacheMemory: no enough memory to repair. need_new_blk_len={}, system = "
          "{},free={},reserved={}",
          need_new_blk_len, query_system_blocks(0.95), get_free_blocks(), get_reserved_blocks());
      throw std::runtime_error("KVCacheMemory: no enough memory to repair");
    }
  }

  for (const auto& mem_state : memory_state_) {
    // SPDLOG_INFO("repair: memory_index={}", mem_state.second.memory_index);
    const auto owned_block_len =
        cached_memories_[mem_state.second.memory_index * 2]->get_block_len();
    if (owned_block_len < mem_state.second.target_block_len) {
      IPIPE_ASSERT(cached_memories_[mem_state.second.memory_index * 2]->repair_by_map(
          mem_state.second.target_block_len));
      IPIPE_ASSERT(cached_memories_[1 + mem_state.second.memory_index * 2]->repair_by_map(
          mem_state.second.target_block_len));
    }
  }
}

}  // namespace kvcache