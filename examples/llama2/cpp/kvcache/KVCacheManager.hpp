#pragma once

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
// #include <torch/torch.h>

#include "KVCacheType.hpp"
#include "KVCacheManagerBase.hpp"

namespace kvcache {
class KVCacheManagerImpl;  // Forward declaration of the implementation class

class KVCacheManager : public KVCacheManagerBase {
 public:
  KVCacheManager();
  virtual ~KVCacheManager();
  KVCacheManager(KVCacheManager&& rhs) = delete;
  KVCacheManager& operator=(KVCacheManager&& rhs) = delete;

  virtual void init(const KVCacheConfig& config);

  /**
   * @return kv cache: layer_num*2
   */
  virtual void alloc_reqid(const KVCacheAllocParams& request_id);

  virtual void free_reqid(const std::string& request_id);

  /**
   * @return kv cahche (num_layers*2) && unused request ids.
   */
  virtual StepOutput step();

 private:
  // const MemOutput& query_key(const std::string& request_id);
  // const MemOutput& query_value(const std::string& request_id);
  // virtual void cache(const CacheParams& ca);
  // QueryOutput query(const std::string& request_id);

 private:
  std::unique_ptr<KVCacheManagerImpl> impl_;  // Pointer to the implementation class
};

}  // namespace kvcache