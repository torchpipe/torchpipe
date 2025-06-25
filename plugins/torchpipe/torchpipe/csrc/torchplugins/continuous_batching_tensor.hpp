#pragma once
#include "hami/core/backend.hpp"
#include "hami/builtin/page_table.hpp"

namespace torchpipe {
using hami::dict;
class TensorPage : public hami::BackendOne {
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const hami::dict& kwargs) {
    page_table_ = &hami::default_page_table();
  }
  void forward(const hami::dict& io) override;

 private:
  hami::PageTable* page_table_{nullptr};
};
} // namespace torchpipe