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

#include "cuda_utils.hpp"
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "ipipe_common.hpp"
namespace ipipe {
std::string get_sm() {
  int device_index = c10::cuda::current_device();
  // c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();

  cudaDeviceProp prop;
  IPIPE_ASSERT(cudaSuccess == cudaGetDeviceProperties(&prop, device_index));

  // int sm_major = prop.major;
  return std::to_string(prop.major) + "." + std::to_string(prop.minor);
}
}  // namespace ipipe