#pragma once
#include <string>
#include <unordered_set>
#include <vector>
#include <memory>
#include <unordered_set>
#include <queue>
#include <atomic>

#include <map>
#include <thread>

#include "threadsafe_queue.hpp"

#include "KVCacheType.hpp"
#include "Memory.hpp"

namespace kvcache {

class CachedMemory {
 public:
  CachedMemory(const KVCacheConfig& config, PyhBlkPool* pyh_pool) {
    // Implementation of the init function
    // Initialize the CachedMemory
    config_ = config;
    pyh_pool_ = pyh_pool;
    // options_ = get_tensor_options(elemsize);
    context_max_size_ = config.elemsize * config.max_seq_len * config.hidden_size;

    seq_per_block_ = config.granularitySize / (config.hidden_size * config.elemsize);
    phy_blocks_.resize(config.layer_num);
    char* k_ptr = (char*)virtual_alloc(context_max_size_ * config.layer_num);
    if (!k_ptr) {
      throw std::runtime_error("Failed to allocate memory");
    }
    for (size_t i = 0; i < config.layer_num; i++) {
      ptrs_.push_back(k_ptr + i * context_max_size_);
      //   auto torch_tensor =
      //       torch::from_blob(k_ptr, {1, config.max_seq_len, config.hidden_size}, options_);
      //   IPIPE_ASSERT(torch_tensor.is_contiguous());
      // k_tensors.push_back(torch_tensor);
    }

    // phy_blocks_[0].resize(config_.granularitySize / (config_.hidden_size * config_.elemsize));
    // 256 blocks maximum
  }

  void offload2cpu(size_t seq_len_with_output);

  void onload2gpu(size_t seq_len_with_output, size_t target_blk_len);

  void release();
  ~CachedMemory() { release(); }
  // void set_target_seq_len(size_t seq_len){

  // }

  const std::vector<char*>& get_ptrs() { return ptrs_; }

  size_t get_block_len() { return phy_blocks_[0].size(); }
  // void set_target_seq_len(size_t seq_len) { target_seq_len_ = seq_len; }
  bool repair_by_map(size_t needed_blk);
  // unextend, 收缩
  bool repair_by_unmap(size_t needed_blk);

 private:
  KVCacheConfig config_;
  // torch::TensorOptions options_;
  std::vector<char*> ptrs_;
  void* cpu_ptr_{nullptr};
  std::vector<std::vector<std::shared_ptr<PhyBlock>>> phy_blocks_;
  size_t seq_per_block_{1};
  // size_t target_seq_len_ = 0;
  // std::vector<size_t> k_mapped_seq_len;
  PyhBlkPool* pyh_pool_{nullptr};
  size_t context_max_size_;
};

class KVCacheMemory {
 public:
  struct CachedMemoryTasks {
    size_t alloc_blks = 0;
    size_t free_reserved = 0;
    std::unordered_set<std::string> offload2cpu;
    std::unordered_set<std::string> onload2gpu;
  };
  KVCacheMemory() = default;
  void wait() {
    while (bInited_.load()) {
      if (task_queue_.WaitEmpty(300)) {
        return;
      }
    }
  }

  const MemOutput& query_key(const std::string& request_id);
  const MemOutput& query_value(const std::string& request_id);

  size_t get_target_block_len(size_t seq_len) {
    const auto seq_per_block = config_.granularitySize / (config_.elemsize * config_.hidden_size);
    auto re = seq_len / seq_per_block;
    if (seq_len % seq_per_block != 0) {
      return re + 1;
    }
    return re;
  }
  size_t get_best_match(int blk_num);
  // sync
  // std::unordered_set<std::string> prefill_alloc_reqs;
  // std::unordered_set<std::string> decode_alloc_reqs;

  // void alloc(const std::unordered_set<std::string>& prefill_alloc_reqs,
  //            const std::unordered_set<std::string>& decode_alloc_reqs);
  void alloc(std::unordered_set<std::string> prefill_alloc_reqs,
             const std::map<std::string, KVCacheState>& prefill_status,
             std::unordered_set<std::string> decode_alloc_reqs,
             const std::map<std::string, KVCacheState>& decode_status);

  size_t alloc_blocks_up_to(size_t blk_size) { return pyh_pool_->alloc(blk_size); }

  void init(const KVCacheConfig& config);
  // void alloc_reqid(const KVCacheAllocParams& request_id);
  void free_reqid(const std::string& request_id) {
    auto func = [this, request_id]() {
      // std::lock_guard<std::mutex> lock(memory_state_mutex_);
      auto iter = memory_state_.find(request_id);
      if (iter != memory_state_.end()) {
        reserved_memory_index_.insert(iter->second.memory_index);
        iter->second.target_block_len = 0;
        memory_state_.erase(iter);
      }
    };
    task_queue_.Push(func);
  }
  void step(const CachedMemoryTasks& tasks);

  // void async_offload2cpu(const std::unordered_set<std::string>& offload2cpu_reqs);
  // void async_onload2gpu();
  // void async_free_reserved(size_t blk_size);
  // bool async_alloc_blks(size_t blk_size);
  int get_free_blocks() { return pyh_pool_->size(); }
  int compute_system_blocks() {
    return pyh_pool_->get_system_free_memory() / (config_.granularitySize);
  }

  size_t query_system_blocks(double factor) {
    return pyh_pool_->query_system_free_memory(factor) / (config_.granularitySize);
  }

  int get_reserved_blocks();
  // void step();
  void extend(size_t len) {
    size_t curr_index = cached_memories_.size() / 2;
    for (size_t i = 0; i < len; i++) {
      auto k_cache = std::make_shared<CachedMemory>(config_, pyh_pool_.get());
      auto v_cache = std::make_shared<CachedMemory>(config_, pyh_pool_.get());
      cached_memories_.push_back(k_cache);
      cached_memories_.push_back(v_cache);
      free_memory_index_.insert(curr_index++);
    }
  }
  ~KVCacheMemory() {
    if (bInited_.load()) {
      release();
    }
  }

  void release() {
    bInited_.store(false);

    if (thread_.joinable()) {
      thread_.join();
    }
    for (auto& memory : cached_memories_) {
      memory->release();
    }
    cached_memories_.clear();
  }

  void cache(const CacheParams& ca);

 private:
  void task_loop();

 private:
  struct KVCacheMemoryState {
    size_t memory_index = 0;
    size_t target_block_len = 0;
    size_t seq_len_with_output = 0;
    // size_t owned_block_len = 0;
    bool offloaded = false;
  };

  void repair();

  int free_reserved(size_t need_free);

  KVCacheConfig config_;
  ipipe::ThreadSafeQueue<std::function<void()>> task_queue_;
  std::vector<std::shared_ptr<CachedMemory>> cached_memories_;

  std::thread thread_;
  std::atomic_bool bInited_{true};
  size_t step_index_{0};
  std::unordered_set<size_t> free_memory_index_;
  std::unordered_set<size_t> reserved_memory_index_;

  std::map<std::string, KVCacheMemoryState> memory_state_;
  // std::mutex memory_state_mutex_;

  std::unique_ptr<PyhBlkPool> pyh_pool_;

  size_t seq_per_block_;
};
}  // namespace kvcache