#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <stdexcept>
#include <chrono>
#include <functional>

namespace hami::queue {

class QueueException : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

class QueueEmptyException : public QueueException {
   public:
    QueueEmptyException() : QueueException("Queue is empty") {}
    using QueueException::QueueException;
};

class QueueFullException : public QueueException {
   public:
    QueueFullException() : QueueException("Queue is full") {}
    using QueueException::QueueException;
};

template <typename T>
class SizedQueue {
   private:
    struct SizedElement {
        T value;
        size_t size;

        SizedElement(const T& val, size_t sz = 1) : value(val), size(sz) {}
    };

    std::queue<SizedElement> queue_;
    size_t totalSize_ = 0;
    size_t maxSize_;

   public:
    explicit SizedQueue(size_t maxSize = 0) : maxSize_(maxSize) {}

    // Push an element with a specified size
    void push(const T& value, size_t size = 1) {
        if (maxSize_ > 0 && totalSize_ + size > maxSize_) {
            throw QueueFullException();
        }
        queue_.push(SizedElement(value, size));
        totalSize_ += size;
    }

    // Pop the front element and reduce the total size
    void pop() {
        if (queue_.empty()) {
            throw QueueEmptyException();
        }
        totalSize_ -= queue_.front().size;
        queue_.pop();
    }

    // Get the front element
    std::pair<const T&, size_t> front() const {
        if (queue_.empty()) {
            throw QueueEmptyException();
        }
        return std::pair<const T&, size_t>(queue_.front().value,
                                           queue_.front().size);
    }

    // Get the total size of the queue
    size_t size() const { return totalSize_; }

    // Check if the queue is empty
    bool empty() const { return queue_.empty(); }

    // Check if the queue is full
    bool full() const { return maxSize_ > 0 && totalSize_ >= maxSize_; }
};

template <typename T>
class ThreadSafeQueue {
   private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    size_t maxsize_;

   public:
    explicit ThreadSafeQueue(size_t maxsize = 0) : maxsize_(maxsize) {}

    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    void put(U&& item, bool block = true,
             std::optional<double> timeout = std::nullopt) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (maxsize_ > 0) {
            if (!block) {
                if (queue_.size() >= maxsize_) {
                    throw QueueFullException();
                }
            } else {
                auto predicate = [this] { return queue_.size() < maxsize_; };

                if (timeout) {
                    if (!cond_.wait_for(lock,
                                        std::chrono::duration<double>(*timeout),
                                        predicate)) {
                        throw QueueFullException("Queue is full after timeout");
                    }
                } else {
                    cond_.wait(lock, predicate);
                }
            }
        }

        queue_.push(std::forward<U>(item));
        cond_.notify_all();
    }

    template <typename Rep, typename Period>
    bool try_put(const T& item, std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (maxsize_ > 0 && queue_.size() >= maxsize_) {
            auto predicate = [this] { return queue_.size() < maxsize_; };
            if (!cond_.wait_for(lock, timeout, predicate)) {
                return false;
            }
        }

        queue_.push(item);
        cond_.notify_all();
        return true;
    }

    T get(bool block = true, std::optional<double> timeout = std::nullopt) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.empty()) {
            if (!block) {
                throw QueueEmptyException();
            } else {
                auto predicate = [this] { return !queue_.empty(); };

                if (timeout) {
                    if (!cond_.wait_for(lock,
                                        std::chrono::duration<double>(*timeout),
                                        predicate)) {
                        throw QueueEmptyException(
                            "Queue is empty after timeout");
                    }
                } else {
                    cond_.wait(lock, predicate);
                }
            }
        }

        T item = std::move(queue_.front());
        queue_.pop();
        // cond_.notify_one();
        return item;
    }

    template <typename Rep, typename Period>
    std::optional<T> try_get(std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!cond_.wait_for(lock, timeout,
                            [this] { return !queue_.empty(); })) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        // cond_.notify_one();
        return item;
    }

    template <typename Rep, typename Period>
    bool wait_for(std::function<bool(size_t)> size_condition,
                  std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        return cond_.wait_for(lock, timeout, [this, size_condition]() {
            return size_condition(queue_.size());
        });
    }

    void notify_one() { cond_.notify_one(); }

    void notify_all() { cond_.notify_all(); }

    size_t size() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    template <template <typename> class Container>
    void force_put_without_notify(const Container<T>& values) {
        std::unique_lock<std::mutex> lock(mutex_);

        for (const auto& value : values) {
            queue_.push(value);
        }
    }

    bool full() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return maxsize_ > 0 && queue_.size() >= maxsize_;
    }
};

template <typename T>
class ThreadSafeSizedQueue {
   private:
    struct SizedElement {
        T value;
        size_t size;

        SizedElement(const T& val, size_t sz = 1) : value(val), size(sz) {}
    };

    std::queue<SizedElement> queue_;
    size_t totalSize_ = 0;
    size_t maxSize_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;

   public:
    explicit ThreadSafeSizedQueue(size_t maxSize = 0) : maxSize_(maxSize) {}

    // Push an element with a specified size
    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    void put(U&& value, size_t size = 1, bool block = true,
             std::optional<double> timeout = std::nullopt) {
        {
            std::unique_lock<std::mutex> lock(mutex_);

            if (maxSize_ > 0) {
                if (!block) {
                    if (totalSize_ + size > maxSize_) {
                        throw QueueFullException();
                    }
                } else {
                    auto predicate = [this, size] {
                        return totalSize_ + size <= maxSize_;
                    };

                    if (timeout) {
                        if (!cond_.wait_for(
                                lock, std::chrono::duration<double>(*timeout),
                                predicate)) {
                            throw QueueFullException(
                                "Queue is full after timeout");
                        }
                    } else {
                        cond_.wait(lock, predicate);
                    }
                }
            }

            queue_.push(SizedElement(std::forward<U>(value), size));
            totalSize_ += size;
        }
        cond_.notify_all();
    }

    std::pair<T, size_t> get(bool block = true,
                             std::optional<double> timeout = std::nullopt) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.empty()) {
            if (!cond_.wait_for(lock, std::chrono::duration<double>(*timeout),
                                [this] { return !queue_.empty(); })) {
                throw QueueEmptyException();
            }
        }

        auto item = std::make_pair(std::move(queue_.front().value),
                                   queue_.front().size);
        totalSize_ -= item.second;
        queue_.pop();
        // cond_.notify_one();
        return item;
    }

    // Get the front element
    std::pair<const T&, size_t> front() const {
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.empty()) {
            throw QueueEmptyException();
        }

        return std::pair<const T&, size_t>(queue_.front().value,
                                           queue_.front().size);
    }

    // Get the total size of the queue
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return totalSize_;
    }

    // Check if the queue is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    // Check if the queue is full
    bool full() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return maxSize_ > 0 && totalSize_ >= maxSize_;
    }

    // please note that there is no notify when get() called. So call it
    // by yourself.
    template <typename Rep, typename Period>
    bool wait_for(std::function<bool(size_t)> size_condition,
                  std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        auto predicate = [this, size_condition]() {
            return size_condition(totalSize_);
        };
        return cond_.wait_for(lock, timeout, predicate);
    }

    // Try to push an element with a specified size within a timeout
    template <typename Rep, typename Period>
    bool try_put(const T& value, size_t size,
                 std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (maxSize_ > 0 && totalSize_ + size > maxSize_) {
            auto predicate = [this, size] {
                return totalSize_ + size <= maxSize_;
            };
            if (!cond_.wait_for(lock, timeout, predicate)) {
                return false;
            }
        }

        queue_.push(SizedElement(value, size));
        totalSize_ += size;
        cond_.notify_all();
        return true;
    }

    template <template <typename> class Container>
    void force_put_without_notify(const Container<T>& values,
                                  size_t size_per_item = 1) {
        std::unique_lock<std::mutex> lock(mutex_);
        for (const auto& value : values) {
            queue_.push(SizedElement(value, size_per_item));
            totalSize_ += size_per_item;
        }
    }

    void notify_one() { cond_.notify_one(); }
    void notify_all() { cond_.notify_all(); }
    // Try to pop an element within a timeout
    template <typename Rep, typename Period>
    std::pair<std::optional<T>, size_t> try_get(
        std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.empty()) {
            auto predicate = [this] { return !queue_.empty(); };
            if (!cond_.wait_for(lock, timeout, predicate)) {
                return std::pair<std::optional<T>, size_t>(std::nullopt, 0);
            }
        }

        auto item = std::pair<std::optional<T>, size_t>(
            std::move(queue_.front().value), queue_.front().size);
        totalSize_ -= item.second;
        queue_.pop();
        // cond_.notify_one();
        return item;
    }
};

}  // namespace hami::queue