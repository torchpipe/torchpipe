#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>

#include "BS_thread_pool.hpp"

namespace om::thread_pool {

BS::thread_pool<>& default_thread_pool(
    const std::string& tag = "",
    size_t size = 0);
} // namespace om::thread_pool