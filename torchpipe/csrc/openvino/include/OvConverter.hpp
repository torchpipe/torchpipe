#pragma once

#include "ModelConverter.hpp"

#include "params.hpp"

namespace ipipe {
namespace model {
struct ModelObject;
class OvConverter : public ModelConverter {
 public:
  OvConverter() = default;
  virtual ~OvConverter() = default;
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param) override;
  virtual std::unique_ptr<ModelInstance> createInstance() override;
  virtual std::vector<std::string> get_input_names() override;
  virtual std::vector<std::string> get_output_names() override;

 private:
  std::unique_ptr<Params> params_;
  std::shared_ptr<ModelObject> model_;
};

}  // namespace model
}  // namespace ipipe