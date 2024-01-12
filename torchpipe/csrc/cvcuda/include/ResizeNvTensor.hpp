#pragma once
#include "Backend.hpp"
#include "params.hpp"
#include <cvcuda/OpResize.hpp>

namespace ipipe {

class ResizeNvTensor : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;
  /**
   * @param TASK_DATA_KEY nvcv::Tensor, 支持hwc
   * @param[out] TASK_RESULT_KEY nvcv::Tensor.
   */
  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  int resize_h_ = 0;
  int resize_w_ = 0;
  cvcuda::Resize ResizeOp_;
};
}  // namespace ipipe