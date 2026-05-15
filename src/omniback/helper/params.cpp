#include "omniback/helper/params.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
namespace om {

void Params::impl_init(
    const std::unordered_map<std::string, std::string>& config) {
  // Process optional parameters with defaults
  for (const auto& [key, default_value] : init_optional_params_) {
    auto it = config.find(key);
    config_[key] = (it != config.end()) ? it->second : default_value;
  }

  // Process required parameters
  for (const auto& req : init_required_params_) {
    OMNI_ASSERT(!req.empty());
    auto it = config.find(req);
    if (it == config.end()) {
      std::string node_name;
      auto it_name = config.find("node_name");
      if (it_name != config.end()) {
        node_name = it_name->second + ": ";
      }
      SPDLOG_ERROR("{}{}: param not set : {}", node_name, req, req);
      throw std::invalid_argument(
          "Params: Incomplete configuration: missing required parameter: " + req);
    }
    config_[req] = it->second;
  }
}
std::string& Params::at(const std::string& key) {
  auto iter = config_.find(key);
  OMNI_ASSERT(iter != config_.end(), "Params: key not found: " + key);
  return iter->second;
}

} // namespace om