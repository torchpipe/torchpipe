#pragma once
#include "omniback/builtin/page_table.hpp"
#include "omniback/core/backend.hpp"

namespace torchpipe {
using om::dict;
class TensorPage : public om::BackendOne {
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const om::dict& kwargs) {
    page_table_ = &om::default_page_table();
  }
  void forward(const om::dict& io) override;

 private:
  om::PageTable* page_table_{nullptr};
};
} // namespace torchpipe