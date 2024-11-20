#pragma once
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <future>
#include <map>

#include "KVCacheType.hpp"
#include "kvcache_utils.hpp"

#include "KVCacheMemory.hpp"
//

namespace kvcache {
// class KVCacheMemoryManager {
//  public:
//   void async_offload2cpu(const std::unordered_set<std::string>& offload2cpu_reqs);
//   void async_onload2gpu();
//   void async_free_reserved(size_t blk_size);
//   bool async_alloc_blks(size_t blk_size);

//   void step();

//   void prefill_alloc(const std::string& request_id);
//   void decode_alloc(const std::string& request_id);

//  private:
// };

class KVCacheSchedule {
 public:
  KVCacheSchedule() = default;
  ~KVCacheSchedule() = default;
  void init(const KVCacheConfig& config);
  void alloc_reqid(const KVCacheAllocParams& request_id);
  void free_reqid(const std::string& request_id);

  StepOutput step();

  // void cache(const CacheParams& ca);

 private:
  std::unordered_set<std::string> valid_prefill_requests(size_t valid_blocks, size_t max_batch_size,
                                                         size_t max_concurrent_requests);
  std::unordered_set<std::string> cal_offload2cpu(const std::unordered_set<std::string>& valid_blks,
                                                  size_t blk_size);
  void prepare_next_memory(KVCacheMemory::CachedMemoryTasks& tasks);
  void prepare_cpu_offload(KVCacheMemory::CachedMemoryTasks& tasks);

  void activate(const std::string& request_id);
  // void activate_decode(const std::string& request_id);

  KVCacheConfig config_;
  std::map<std::string, KVCacheState> decode_kvcache_;
  std::map<std::string, KVCacheState> prefill_kvcache_;
  std::map<std::string, KVCacheState> offload2cpu_kvcache_;
  std::map<std::string, KVCacheState> onload2gpu_kvcache_;
  // size_t step_index_{0};

  size_t seq_per_block_{1};
  std::unique_ptr<KVCacheMemory> memory_;
  std::unordered_set<std::string> valid_prefill_reqs_;
  std::unordered_set<std::string> valid_decode_reqs_;

  size_t min_blk_ = 0;
  int system_blocks_ = INT_MIN;

  bool need_update_system_blk_ = true;

  // std::queue<std::vector<torch::Tensor>>
  //     k_tensors_;  // key_input, value_input, key_output, value_output ... 32
  // std::queue<std::vector<torch::Tensor>> v_tensors_;
};
}  // namespace kvcache