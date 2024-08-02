#include "SyncRing.hpp"

#include "event.hpp"
#include "ipipe_common.hpp"
#include "config_parser.hpp"
#include "dict_helper.hpp"
#include "base_logging.hpp"

namespace ipipe {

bool SyncRing::init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"SyncRing::backend", "PipelineV3"}}, {}, {}, {}));
  if (!params_->init(config)) return false;

  //   IPIPE_ASSERT(dict_config);
  //   auto iter_config = dict_config->find("config");
  //   IPIPE_ASSERT(iter_config != dict_config->end());
  //   mapmap config_param = any_cast<mapmap>(iter_config->second);
  //   ring_ = handle_ring(config_param);

  auto iter = dict_config->find("backend");
  if (iter != dict_config->end()) {
    backend_ = any_cast<Backend*>(iter->second);
    assert(backend_);
  } else {
    owned_backend_ =
        std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("SyncRing::backend")));
    if (!owned_backend_ || !owned_backend_->init(config, dict_config)) return false;
    backend_ = owned_backend_.get();
  }

  return true;
}

void SyncRing::forward(const std::vector<dict>& input_dicts) {
  //   const std::string first_node_name =
  //   any_cast<std::string>(input_dicts.at(0)->at("node_name"));

  //   for (size_t i = 0; i < input_dicts.size(); i++) {
  //     auto iter = item->find(TASK_EVENT_KEY);
  //     IPIPE_ASSERT(iter == item->end());
  //     if (i > 0) {
  //       std::string node_name = any_cast<std::string>(item->at("node_name"));
  //       if (node_name != first_node_name) {
  //         throw std::runtime_error("All node_names should be the same");
  //       }
  //     }
  //   }
  {
    // DictHelper guard(input_dicts);
    // guard.keep("node_name");
    for (size_t i = 0; i < input_dicts.size(); i++) {
      const auto& item = input_dicts[i];
      // IPIPE_ASSERT(item->find(TASK_EVENT_KEY) == item->end());
    }
    backend_->forward(input_dicts);
  }

  do {
    std::vector<dict> restart_dicts;
    for (size_t i = 0; i < input_dicts.size(); i++) {
      const auto& item = input_dicts[i];
      // IPIPE_ASSERT(item->find(TASK_EVENT_KEY) == item->end());
      auto iter = item->find(TASK_RESTART_KEY);
      if (iter != item->end()) {
        std::string restart_node_name = any_cast<std::string>(iter->second);
        restart_dicts.push_back(item);
        item->erase(iter);
        (*item)["node_name"] = restart_node_name;
        SPDLOG_DEBUG("RESTART: " + restart_node_name);
      }
    }
    if (!restart_dicts.empty()) {
      backend_->forward(restart_dicts);
    } else {
      break;
    }
  } while (true);
}

IPIPE_REGISTER(Backend, SyncRing, "SyncRing");
}  // namespace ipipe
