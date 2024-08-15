
#include <algorithm>

#include "Composite.hpp"
#include "base_logging.hpp"
#include "exception.hpp"
#include "params.hpp"
namespace ipipe {
bool Composite::init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
  std::string class_name_backend = IPIPE_GET_REGISTER_NAME(Backend, Composite, this) + "::backend";
  // Composite::backend or C::backend
  params_ = std::unique_ptr<Params>(new Params({{class_name_backend, "Ring"}}, {}, {}, {}));
  if (!params_->init(config)) return false;

  std::vector<std::string> engine_names_ =
      str_split_brackets_match(params_->at(class_name_backend), ',', '[', ']');
  IPIPE_ASSERT(!engine_names_.empty());

  std::vector<std::string> final_frontend_engine_names;
  std::vector<std::string> final_backend_engine_names;

  std::string start_frontend = engine_names_[0];

  std::unordered_map<std::string, std::string> final_config = config;

  for (size_t i = 0; i < engine_names_.size(); ++i) {
    const auto& engine = engine_names_[i];
    std::unordered_map<std::string, std::string> new_config;

    brackets_split(engine, new_config);
    for (const auto& item : new_config) {
      final_config[item.first] = item.second;
    }

    auto iter = new_config.find(engine + "::backend");
    std::string final_backend = engine;
    while (true) {
      if (iter != new_config.end()) {
        final_backend = iter->second;
      } else {
        auto tmp = IPIPE_GET_DEFAULT_BACKEND(final_backend);
        if (!tmp.empty()) {
          final_backend = tmp;
        } else {
          break;
        }
      }
      iter = new_config.find(final_backend + "::backend");
    }

    final_backend_engine_names.push_back(final_backend);

    auto front = IPIPE_GET_DEFAULT_FRONTEND(engine);
    if (i == 0 && !front.empty()) {
      start_frontend = front;
    }
    final_frontend_engine_names.push_back(front);

    // engines_.emplace_back(IPIPE_CREATE(Backend, new_config.at("backend")));
  }

  for (size_t i = 0; i < engine_names_.size(); ++i) {
    if (!final_frontend_engine_names[i].empty()) {
      auto origin_name = engine_names_[i];
      engine_names_[i] = final_frontend_engine_names[i];
      final_config[final_frontend_engine_names[i] + "::backend"] = origin_name;
      // todo : only support  one frontend now;
      IPIPE_ASSERT(IPIPE_GET_DEFAULT_FRONTEND(final_frontend_engine_names[i]).empty());
    }
  }

  for (size_t i = 0; i < engine_names_.size() - 1; ++i) {
    IPIPE_ASSERT(!final_backend_engine_names[i].empty());
    final_config[final_backend_engine_names[i] + "::backend"] = engine_names_[i + 1];
  }

  backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, start_frontend));

  IPIPE_ASSERT(backend_ && backend_->init(final_config, dict_config));

  return true;
}

void Composite::forward(const std::vector<dict>& inputs) { backend_->forward(inputs); }

IPIPE_REGISTER(Backend, Composite, "Composite,C");

}  // namespace ipipe