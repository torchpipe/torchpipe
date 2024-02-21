#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

#include "any.hpp"

namespace ipipe {
namespace model {

class ModelInstance {
 public:
  virtual void forward() = 0;

  virtual void set_input(const std::string& name, const any& data) = 0;
  virtual any get_output(const std::string& name) = 0;
  virtual std::vector<int> get_shape(const std::string& name) { return {}; };

 public:
  virtual ~ModelInstance() = default;

 private:
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param) {
    return true;
  }

  friend class ModelConverter;
};

class ModelConverter {
 public:
  ModelConverter() = default;
  virtual ~ModelConverter(){};
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param) = 0;
  virtual std::unique_ptr<ModelInstance> createInstance() = 0;
  virtual std::vector<std::string> get_input_names() = 0;
  virtual std::vector<std::string> get_output_names() = 0;
};

}  // namespace model
}  // namespace ipipe