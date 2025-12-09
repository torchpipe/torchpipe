#include "omniback/helper/params.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
namespace omniback {

void Params::impl_init(
    const std::unordered_map<std::string, std::string>& config) {
  for (auto iter = init_optional_params_.begin();
       iter != init_optional_params_.end();
       ++iter) {
    auto iter_config = config.find(iter->first);
    if (iter_config == config.end()) {
      config_[iter->first] = iter->second;
    } else {
      config_[iter->first] = iter_config->second;
    }
  }

  for (const auto& req : init_required_params_) {
    OMNI_ASSERT(!req.empty());
    auto iter_config = config.find(req);
    if (iter_config == config.end()) {
      std::string node_name;
      auto iter_name = config.find("node_name");
      node_name = (iter_name == config.end()) ? "" : iter_name->second + ": ";
      SPDLOG_ERROR(node_name + ": param not set : " + req);
      throw std::invalid_argument(
          "Params: Incomplete configuration: missing required parameter");
    } else {
      config_[req] = iter_config->second;
    }
  }
}
std::string& Params::at(const std::string& key) {
  auto iter = config_.find(key);
  OMNI_ASSERT(iter != config_.end(), "Params: key not found: " + key);
  return iter->second;
}

} // namespace omniback