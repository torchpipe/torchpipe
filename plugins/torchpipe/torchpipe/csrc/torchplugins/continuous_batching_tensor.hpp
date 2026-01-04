#pragma once
#include "omniback/builtin/page_table.hpp"
#include "omniback/core/backend.hpp"

namespace torchpipe {
using omniback::dict;
class TensorPage : public omniback::BackendOne {
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const omniback::dict& kwargs) {
    page_table_ = &omniback::default_page_table();
  }
  void forward(const omniback::dict& io) override;

 private:
  omniback::PageTable* page_table_{nullptr};
};
} // namespace torchpipe