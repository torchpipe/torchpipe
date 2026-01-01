// resource_pool.hpp
#ifndef RESOURCE_MANAGER_RESOURCE_POOL_HPP
#define RESOURCE_MANAGER_RESOURCE_POOL_HPP

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace omniback::pool {

class pool_exhausted : public std::runtime_error {
 public:
  explicit pool_exhausted(const char* msg) : std::runtime_error(msg) {}
};

class invalid_resource : public std::logic_error {
 public:
  explicit invalid_resource(const char* msg) : std::logic_error(msg) {}
};

template <typename IdType>
class ResourcePool final {
 public:
  using id_type = IdType;
  using size_type = std::size_t;

  explicit ResourcePool(id_type max_id)
      : capacity_(static_cast<size_type>(max_id)),
        available_count_(capacity_),
        free_ids_(capacity_) {
    for (id_type i = 0; i < max_id; ++i) {
      free_ids_[i] = i;
    }
  }

  ~ResourcePool() = default;

  ResourcePool(const ResourcePool&) = delete;
  ResourcePool& operator=(const ResourcePool&) = delete;
  ResourcePool(ResourcePool&&) = delete;
  ResourcePool& operator=(ResourcePool&&) = delete;

  id_type acquire() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return available_count_ > 0; });

    id_type id = free_ids_[--available_count_];
    return id;
  }

  bool try_acquire(id_type& id) noexcept {
    if (available_count_.load(std::memory_order_relaxed) == 0) {
      return false;
    }

    std::lock_guard lock(mutex_);
    if (available_count_ == 0) {
      return false;
    }

    id = free_ids_[--available_count_];
    return true;
  }

  void release(id_type id) {
    const size_type idx = static_cast<size_type>(id);
    if (idx >= capacity_) {
      throw invalid_resource("Attempted to release invalid resource ID");
    }

    {
      std::lock_guard lock(mutex_);

      free_ids_[available_count_++] = id;
    }

    cv_.notify_one();
  }

  size_type available() const noexcept {
    return available_count_.load(std::memory_order_relaxed);
  }

  size_type used() const noexcept {
    return capacity_ - available_count_.load(std::memory_order_relaxed);
  }

  size_type capacity() const noexcept {
    return capacity_;
  }

  class lease_guard {
   public:
    explicit lease_guard(ResourcePool* pool, id_type id) noexcept
        : pool_(pool), id_(id), released_(false) {}

    ~lease_guard() {
      if (!released_.load(std::memory_order_acquire)) {
        try {
          pool_->release(id_);
        } catch (...) {
          // Swallow exceptions in destructor
        }
      }
    }

    id_type get() const noexcept {
      return id_;
    }

    void early_release() {
      if (!released_.exchange(true, std::memory_order_release)) {
        pool_->release(id_);
      }
    }

    lease_guard(const lease_guard&) = delete;
    lease_guard& operator=(const lease_guard&) = delete;
    lease_guard(lease_guard&&) = delete;
    lease_guard& operator=(lease_guard&&) = delete;

   private:
    ResourcePool* pool_;
    id_type id_;
    std::atomic<bool> released_;
  };

  lease_guard acquire_with_lease_guard() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return available_count_ > 0; });

    id_type id = free_ids_[--available_count_];
    return lease_guard(this, id);
  }

 private:
  const size_type capacity_;
  std::atomic<size_type> available_count_;
  std::vector<id_type> free_ids_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
};

} // namespace omniback::pool

#endif // RESOURCE_MANAGER_RESOURCE_POOL_HPP