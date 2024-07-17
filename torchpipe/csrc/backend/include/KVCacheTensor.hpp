// Copyright 2021-2024 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "Backend.hpp"
#include "dict.hpp"

#include <memory>

namespace ipipe {
class Params;

class KVCacheTensor : public SingleBackend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  virtual void forward(dict) override;

 private:
  std::unique_ptr<Params> params_;
  std::unique_ptr<Backend> engine_;
};
}  // namespace ipipe
