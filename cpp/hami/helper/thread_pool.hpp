#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <stdexcept>
#include <chrono>
#include <functional>

#include "BS_thread_pool.hpp"

namespace hami::thread_pool {

BS::thread_pool<>& default_thread_pool(const std::string& tag = "",
                                       size_t size = 0);
}  // namespace hami::thread_pool