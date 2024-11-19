#pragma once

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <torch/torch.h>

#include "KVCacheType.hpp"
#include "KVCacheSchedule.hpp"
// #include "KVCacheMemory.hpp"

namespace kvcache {

class KVCacheManagerImpl {
 public:
  ~KVCacheManagerImpl();

  void init(const KVCacheConfig& config);

  void alloc_reqid(const KVCacheAllocParams& request_id);

  void free_reqid(const std::string& request_id);
  // const MemOutput& query_key(const std::string& request_id);
  // const MemOutput& query_value(const std::string& request_id);
  // void cache(const CacheParams& ca);
  StepOutput step();

 private:
  // Private members for managing the cache
  // std::unordered_map<std::string, std::vector<std::pair<torch::Tensor, torch::Tensor>>> cache_;
  // size_t step_index_{0};
  std::unique_ptr<KVCacheSchedule> schedule_;
  // std::unique_ptr<KVCacheMemory> memory_;
};

}  // namespace kvcache