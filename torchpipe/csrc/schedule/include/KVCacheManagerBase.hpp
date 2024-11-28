#pragma once

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
// #include <torch/torch.h>

#include "KVCacheType.hpp"

namespace kvcache {

class KVCacheManagerBase {
 public:
  KVCacheManagerBase() = default;
  virtual ~KVCacheManagerBase() = default;
  KVCacheManagerBase(KVCacheManagerBase&& rhs) = delete;
  KVCacheManagerBase& operator=(KVCacheManagerBase&& rhs) = delete;

  virtual void init(const KVCacheConfig& config) {}

  /**
   * @return kv cache: layer_num*2
   */
  virtual void alloc_reqid(const KVCacheAllocParams& request_id) = 0;

  virtual void free_reqid(const std::string& request_id) = 0;

  virtual StepOutput step() = 0;
  // virtual const MemOutput& query_key(const std::string& request_id) = 0;
  // virtual const MemOutput& query_value(const std::string& request_id) = 0;
  // virtual void cache(const CacheParams& params) = 0;
};

}  // namespace kvcache