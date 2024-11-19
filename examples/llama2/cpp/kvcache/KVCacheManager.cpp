#include "KVCacheManager.hpp"
#include <unordered_map>
#include <iostream>
#include "KVCacheManagerImpl.hpp"
#include "reflect.h"

namespace kvcache {
KVCacheManager::KVCacheManager() : impl_(std::make_unique<KVCacheManagerImpl>()) {}
KVCacheManager::~KVCacheManager() {}

void KVCacheManager::init(const KVCacheConfig& config) { impl_->init(config); }

void KVCacheManager::alloc_reqid(const KVCacheAllocParams& request_id) {
  return impl_->alloc_reqid(request_id);
}

void KVCacheManager::free_reqid(const std::string& request_id) { impl_->free_reqid(request_id); }

StepOutput KVCacheManager::step() { return impl_->step(); }

// const MemOutput& KVCacheManager::query_key(const std::string& request_id) {
//   return impl_->query_key(request_id);
// }
// const MemOutput& KVCacheManager::query_value(const std::string& request_id) {
//   return impl_->query_value(request_id);
// }
// QueryOutput KVCacheManager::query(const std::string& request_id) {
//   return impl_->query(request_id);
// }

// void KVCacheManager::cache(const CacheParams& ca) { return impl_->cache(ca); }

// IPIPE_REGISTER(KVCacheManagerBase, KVCacheManager, "KVCacheManager");
static ipipe::reflect::ClassRegisterer<KVCacheManagerBase> KVCacheManagerRegistryTag(
    ipipe::reflect::ClassRegistry_NewObject<KVCacheManagerBase, KVCacheManager>, "KVCacheManager",
    {"KVCacheManager", "KVCacheManager"});

}  // namespace kvcache
