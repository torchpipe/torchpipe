#pragma once
#include "Backend.hpp"
#include "params.hpp"

namespace ipipe {
class CreateTensor : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  std::vector<long int> shape_;
  std::string type_;
};

class AppendPositionIDsTensor : public SingleBackend {
 public:
  virtual void forward(dict) override;

 private:
};

}  // namespace ipipe