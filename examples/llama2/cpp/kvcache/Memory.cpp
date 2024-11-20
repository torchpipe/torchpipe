#include <iostream>
#include <vector>
#include <queue>
#include <set>

#include <string>
#include <atomic>
#include "Memory.hpp"
#include "ipipe_common.hpp"
#include "base_logging.hpp"
#include "cuda_runtime.h"
#include "time_utils.hpp"
// https://github.com/XinYao1994/glake/blob/28046fcffede3a901c1033b1f10089cca68b21cf/vTensor/vllm/worker/worker.py#L176
namespace kvcache {

size_t PyhBlkPool::get_system_free_memory() {
  return system_mem_ - num_allocated_ * block_size_;
  // static auto x = [device = device_id_]() {
  //   IPIPE_ASSERT(cudaSuccess == cudaSetDevice(device));
  //   return true;
  // }();
  // size_t total;
  // // if (system_mem_ == 0) {
  // ipipe::TimeGuard guard("PyhBlkPool::cuMemGetInfo");
  // DRV_CALL(cuMemGetInfo(&system_mem_, &total));
  // // }

  // return system_mem_;
}

size_t PyhBlkPool::query_system_free_memory(double factor) {
  static auto x = [device = device_id_]() {
    IPIPE_ASSERT(cudaSuccess == cudaSetDevice(device));
    return true;
  }();
  size_t total;
  // if (system_mem_ == 0) {
  ipipe::TimeGuard guard("PyhBlkPool::cuMemGetInfo");
  DRV_CALL(cuMemGetInfo(&system_mem_, &total));
  // }
  system_mem_ = system_mem_ * factor;
  num_allocated_ = 0;
  SPDLOG_INFO("PyhBlkPool Guery Result: device_id={} system_mem_ = {}MB", device_id_,
              system_mem_ / 1024 / 1024);

  return system_mem_;
}

PyhBlkPool::PyhBlkPool(int device_id, size_t block_size)
    : device_id_(device_id), block_size_(block_size) {
  // DRV_CALL(cuInit(0));
  IPIPE_ASSERT(device_id_ >= 0);
  cudaSetDevice(device_id_);
  cudaFree(0);

  int deviceCount = 0;
  DRV_CALL(cuDeviceGetCount(&deviceCount));

  CUdevice cu_dev;
  DRV_CALL(cuDeviceGet(&cu_dev, device_id_));

  CUcontext cu_ctx;
  DRV_CALL(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
  SPDLOG_INFO("PyhBlkPool: device_id: {} deviceCount = {} cu_ctx = {}", device_id_, deviceCount,
              (void*)cu_ctx);

  // if (device_id_ == -1) {
  //   DRV_CALL(cuCtxGetDevice(&device_id_));
  // }
  DRV_CALL(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING));
  // get_system_free_memory();
}

PhyBlock::PhyBlock(int device_id, size_t granularitySize)
    : device_id_(device_id), block_size_(granularitySize) {
  accessDesc_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc_.location.id = device_id_;
  accessDesc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

PhyBlock::~PhyBlock() { release(); }

void PhyBlock::map(char* virtual_ptr) {
  assert(virtual_ptr_ == nullptr);
  DRV_CALL(
      cuMemMap(reinterpret_cast<CUdeviceptr>(virtual_ptr), block_size_, 0ULL, alloc_handle_, 0ULL));

  DRV_CALL(
      cuMemSetAccess(reinterpret_cast<CUdeviceptr>(virtual_ptr), block_size_, &accessDesc_, 1));
  virtual_ptr_ = virtual_ptr;
}
void PhyBlock::unmap(char* virtual_ptr) {
  // todo : multiple map
  if (virtual_ptr_ == nullptr) {
    return;
  }
  DRV_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(virtual_ptr_), block_size_));
  virtual_ptr_ = nullptr;
}
}  // namespace kvcache