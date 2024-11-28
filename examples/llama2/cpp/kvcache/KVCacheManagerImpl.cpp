
#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <torch/torch.h>

#include "spdlog/spdlog.h"

#include "KVCacheManagerImpl.hpp"
// #include "KVCacheSchedule.hpp"
// #include "KVCacheMemory.hpp"

namespace kvcache {

KVCacheManagerImpl::~KVCacheManagerImpl() = default;
void KVCacheManagerImpl::init(const KVCacheConfig& config) {
  schedule_ = std::make_unique<KVCacheSchedule>();
  // memory_ = std::make_unique<KVCacheMemory>();
  schedule_->init(config);
  // memory_->init(config);
  return;
}

void KVCacheManagerImpl::alloc_reqid(const KVCacheAllocParams& request_id) {
  // Implementation of the alloc_reqid function
  // std::cout << "Allocating kv cache for request_id: " << request_id << std::endl;
  // Allocate and return kv cache
  return schedule_->alloc_reqid(request_id);
}

void KVCacheManagerImpl::free_reqid(const std::string& request_id) {
  // Implementation of the free_reqid function
  // std::cout << "Freeing kv cache for request_id: " << request_id << std::endl;
  // Free the kv cache
  return schedule_->free_reqid(request_id);
}

StepOutput KVCacheManagerImpl::step() {
  // Implementation of the step function
  // std::cout << "Stepping through KVCacheManagerImpl" << std::endl;
  // Return unused request ids
  return schedule_->step();
  // step_index_++;
  // return {};
}

// const MemOutput& KVCacheManagerImpl::query_key(const std::string& request_id) {
//   return schedule_->query_key(request_id);
// }
// const MemOutput& KVCacheManagerImpl::query_value(const std::string& request_id) {
//   return schedule_->query_value(request_id);
// }

// QueryOutput KVCacheManagerImpl::query(const std::string& request_id) {
//   return schedule_->query(request_id);
// }

// void KVCacheManagerImpl::cache(const CacheParams& ca) { return schedule_->cache(ca); }

}  // namespace kvcache