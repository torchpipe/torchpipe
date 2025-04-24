#include <atomic>

#include <cstdint>

namespace hami {

template <typename T>
class AtomicType {
 public:
  explicit AtomicType(T value = 0) : value(value) {}

  T get() const noexcept {
    return value.load(std::memory_order_acquire);
  }

  void set(T new_value) noexcept {
    value.store(new_value, std::memory_order_release);
  }

  // 原子 += 操作（返回自身引用）
  AtomicType<T>& operator+=(T delta) noexcept {
    value.fetch_add(delta, std::memory_order_acq_rel);
    return *this;
  }

  // 前缀 ++（返回新值引用）
  T operator++() noexcept {
    return value.fetch_add(1, std::memory_order_acq_rel) + 1;
  }

  // 后缀 ++（返回旧值）
  T operator++(int) noexcept {
    return value.fetch_add(1, std::memory_order_acq_rel);
  }

  // 原子比较交换
  bool compare_exchange(T expected, T desired) noexcept {
    return value.compare_exchange_strong(
        expected,
        desired,
        std::memory_order_acq_rel,
        std::memory_order_acquire);
  }

 private:
  std::atomic<T> value;
};

template class AtomicType<int>;
} // namespace hami