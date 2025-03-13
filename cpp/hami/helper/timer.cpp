
#include "hami/helper/timer.hpp"
#include "hami/helper/base_logging.hpp"
#include <chrono>
#include <random>

namespace hami::helper
{
    bool should_log(float probability)
    {
        if (probability >= 1.0)
            return true;
        if (probability <= 0.0)
            return false;

        // 使用线程安全的随机数生成（C++11）
        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        return distribution(generator) <= probability;
    }
    ScopedTimer::~ScopedTimer()
    {
        if (!should_log(probability_))
            return;
        const auto end = std::chrono::steady_clock::now();
        const auto duration = end - start_;
        const double ms = std::chrono::duration<double, std::milli>(duration).count();
        SPDLOG_INFO("[{}] took {} ms\n", name_, ms);
    }
}