#include <atomic>

#include <cstdint>

namespace omniback {

template <typename T>
class AtomicType {
 public:
  AtomicType() : value(T{}) {}
  explicit AtomicType(T val) : value(val) {}

  // 删除拷贝和移动
  AtomicType(const AtomicType&) = delete;
  AtomicType& operator=(const AtomicType&) = delete;

  T get() const noexcept {
    return value.load(std::memory_order_acquire);
  }

  void set(T new_value) noexcept {
    value.store(new_value, std::memory_order_release);
  }

  AtomicType<T>& operator+=(T delta) noexcept {
    value.fetch_add(delta, std::memory_order_acq_rel);
    return *this;
  }

  T operator++() noexcept {
    return value.fetch_add(1, std::memory_order_acq_rel) + 1;
  }

  T operator++(int) noexcept {
    return value.fetch_add(1, std::memory_order_acq_rel);
  }

  bool compare_exchange(T& expected, T desired) noexcept {
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
} // namespace omniback