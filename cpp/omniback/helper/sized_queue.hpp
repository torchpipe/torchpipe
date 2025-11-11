#pragma once

#include <queue>
#include <utility>

namespace omniback {
// Template class for a queue where each element has an associated size.
template <typename T>
class SizedQueue {
 private:
  // The underlying queue. Each element is a pair of a value and its size.
  std::queue<std::pair<T, std::size_t>> queue_;
  // The total size of all elements in the queue.
  std::size_t total_size_ = 0;

 public:
  // Returns true if the queue is empty, false otherwise.
  bool empty() const {
    return queue_.empty();
  }

  // Returns the total size of all elements in the queue.
  std::size_t size() const {
    return total_size_;
  }

  // Adds an element with the given value and size to the queue.
  void push(const T& value, std::size_t size) {
    queue_.push({value, size});
    total_size_ += size;
  }

  // Removes the first element from the queue.
  void pop() {
    // if (!queue_.empty()) {
    total_size_ -= queue_.front().second;
    queue_.pop();
    // }
    // original: undefined for empty queue
  }

  // Returns the value of the first element in the queue.
  T& front() {
    return queue_.front().first;
  }

  std::size_t front_size() const {
    return queue_.front().second;
  }

  // Returns the value of the first element in the queue. This version of the
  // function is const.
  const T& front() const {
    return queue_.front().first;
  }
};
} // namespace omniback