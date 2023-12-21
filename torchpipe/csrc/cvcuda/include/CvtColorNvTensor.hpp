#pragma once
#include "Backend.hpp"
#include "params.hpp"
#include <cvcuda/OpCvtColor.hpp>

namespace ipipe {

class CvtColorNvTensor : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;
  /**
   * @param TASK_DATA_KEY at::Tensor, 支持hwc 和 1hwc(其中 c==3)
   * @param[out] TASK_RESULT_KEY nvcv::Tensor.
   */
  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  std::string color_;
  cvcuda::CvtColor cvtColorOp_;
};
}  // namespace ipipe