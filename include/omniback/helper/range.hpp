#include <iterator>

namespace omniback {
template <typename T>
class range {
 public:
  class Iterator {
   public:
    using value_type = T; // 类型别名
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::input_iterator_tag;

    Iterator(T value) noexcept : value(value) {}
    Iterator& operator++() noexcept {
      ++value;
      return *this;
    }
    Iterator operator++(int) noexcept {
      return Iterator(value++);
    }
    bool operator!=(const Iterator& other) const noexcept {
      return value != other.value;
    }
    bool operator==(const Iterator& other) const noexcept {
      return value == other.value;
    }
    T operator*() const noexcept {
      return value;
    }

   private:
    T value;
  };

  range(T start_param, T end_param) noexcept
      : start_(start_param), end_(end_param) {}
  range(T end_param) noexcept : start_(0), end_(end_param) {}

  Iterator begin() const noexcept {
    return Iterator(start_);
  }
  Iterator end() const noexcept {
    return Iterator(end_);
  }

 private:
  T start_;
  T end_;
};
} // namespace omniback