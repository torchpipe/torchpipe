// Copyright 2021-2023 NetEase.
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

#include "Backend.hpp"
#include "reflect.h"
namespace ipipe {

class Identity : public SingleBackend {
 public:
  void forward(dict input_dict) {
    assert(input_dict->find(TASK_DATA_KEY) != input_dict->end());
    (*input_dict)[TASK_RESULT_KEY] = (*input_dict)[TASK_DATA_KEY];
  }
};

IPIPE_REGISTER(Backend, Identity, "Identity");
using Empty = Identity;
IPIPE_REGISTER(Backend, Empty, "Empty");
}  // namespace ipipe
