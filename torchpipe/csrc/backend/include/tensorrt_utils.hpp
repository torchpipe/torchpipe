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
#include <NvInferRuntime.h>
#include "NvInfer.h"
#include <memory>
#include <set>
#include <string>

#if (NV_TENSORRT_MAJOR >= 9 || (NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >= 6))
#define USE_TORCH_ALLOCATOR
#include "NvInferRuntime.h"
#endif

#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 5) || (NV_TENSORRT_MAJOR >= 11)
#define USER_MANAGED_MEM 1
#else
#define USER_MANAGED_MEM 0
#endif

namespace ipipe {

struct destroy_nvidia_pointer {
  template <class T>
  void operator()(T *obj) const {
    if (obj) {
#if NV_TENSORRT_MAJOR < 8
      obj->destroy();
#else
      delete obj;
#endif
    }
  }
};

template <class T>
using unique_ptr_destroy = std::unique_ptr<T, destroy_nvidia_pointer>;

// support tensorrt8.6.1, which requires that runtime keep alive when engine it is in use.
struct CudaEngineWithRuntime {
  explicit CudaEngineWithRuntime() = default;
  explicit CudaEngineWithRuntime(nvinfer1::IRuntime *runtime_ptr) : runtime(runtime_ptr) {};
  explicit CudaEngineWithRuntime(nvinfer1::ICudaEngine *engine_ptr) : engine(engine_ptr) {};

  bool deserializeCudaEngine(const std::string &engine_plan) {
    if (runtime) {
      engine = runtime->deserializeCudaEngine(engine_plan.data(), engine_plan.size());
      if (engine) return true;
    }
    return false;
  }
  bool deserializeCudaEngine(const void *data, std::size_t len) {
    if (runtime) {
      engine = runtime->deserializeCudaEngine(data, len);
      if (engine) return true;
    }
    return false;
  }

  ~CudaEngineWithRuntime() {
    if (engine) destroy_nvidia_pointer()(engine);
    if (runtime) destroy_nvidia_pointer()(runtime);
    if (allocator) destroy_nvidia_pointer()(allocator);
  }
  nvinfer1::ICudaEngine *engine = nullptr;
  nvinfer1::IRuntime *runtime = nullptr;
  nvinfer1::IGpuAllocator *allocator = nullptr;
};

bool check_dynamic_batchsize(nvinfer1::INetworkDefinition *network);

bool precision_fpx_count(const std::set<std::string> &layers, const std::string &name,
                         std::set<std::string> &layers_erased);
bool is_ln_name(const std::set<std::string> &layers, const std::string &name);
void parse_ln(nvinfer1::INetworkDefinition *network);
void modify_layers_precision(std::set<std::string> precision_fpx,
                             nvinfer1::INetworkDefinition *network, nvinfer1::DataType dataType,
                             bool is_output = false);
nvinfer1::ITensor *MeanStd(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *input,
                           float *mean, float *std, std::set<nvinfer1::ILayer *> &new_layers,
                           bool set_half);
bool is_qat(nvinfer1::INetworkDefinition *network);

}  // namespace ipipe