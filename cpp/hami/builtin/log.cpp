#include "hami/helper/base_logging.hpp"
#include "hami/core/backend.hpp"

#include "spdlog/sinks/stdout_color_sinks.h"

namespace hami {
class DebugLogger : public Backend {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override {
    default_logger()->set_level(spdlog::level::debug);
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    console_sink->set_color_mode(spdlog::color_mode::always);
    default_logger()->sinks().clear();
    default_logger()->sinks().push_back(console_sink);
    default_logger()->set_pattern("[%H:%M:%S.%e] [%l] [%s:%#] %v");
  }
};
HAMI_REGISTER_BACKEND(DebugLogger);
} // namespace hami