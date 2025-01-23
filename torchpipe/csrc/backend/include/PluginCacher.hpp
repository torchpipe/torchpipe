#pragma once

#include "Backend.hpp"
#include "params.hpp"
#include "resource_guard.hpp"

namespace ipipe {
struct BackendConfig {
  std::unordered_map<std::string, std::string> config;
  dict dict_config;
  /* data */
};

class PluginCacher : public Backend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string> &, dict) override;

  virtual void forward(const std::vector<dict> &) override;
  virtual uint32_t min() const { return engine_->min(); };
  virtual uint32_t max() const { return engine_->max(); };

  static const std::vector<dict> &query_input();

  static const std::vector<dict> &query_input(void *stream);
  static const BackendConfig &query_config(void *stream);
  static dict query_dict_config(void *stream);

 private:
  std::unique_ptr<Params> params_;
  std::unique_ptr<Backend> engine_;
  BackendConfig pack_config_param_;

  static ResourceGuard<BackendConfig> config_;
};

class ThreadCacher : public Backend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string> &, dict) override;

  virtual void forward(const std::vector<dict> &) override;
  virtual uint32_t min() const { return engine_->min(); };
  virtual uint32_t max() const { return engine_->max(); };

  static const std::vector<dict> &query_input();

  static const std::unordered_map<std::string, std::string> &query_config();
  static dict query_dict_config();
  static bool input_cached() { return input_dicts_ != nullptr; }

 private:
  std::unique_ptr<Params> params_;
  std::unique_ptr<Backend> engine_;

  static thread_local std::unordered_map<std::string, std::string> config_;
  static thread_local dict dict_config_;

  static thread_local const std::vector<dict> *input_dicts_;
};
}  // namespace ipipe